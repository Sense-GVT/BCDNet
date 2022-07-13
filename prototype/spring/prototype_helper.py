# Standard Library
import os
import copy
import time
import json
import yaml
import shutil
import datetime
from easydict import EasyDict
import pprint
import numpy as np
# Import from third library
import torch
from tensorboardX import SummaryWriter
import linklink as link
import torch.nn.functional as F
# import from prototype
from prototype.utils.dist import DistModule, broadcast_object
from prototype.utils.misc import ( # noqa
    makedir, create_logger, load_state_model, get_logger, count_params, count_flops,
    param_group_all, AverageMeter, accuracy, modify_state, load_state_optimizer
)
from prototype.utils.ema import EMA
from prototype.utils.nnie_helper import generate_nnie_config
from prototype.model import model_entry
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD # noqa
from prototype.lr_scheduler import scheduler_entry
from prototype.loss_functions import LabelSmoothCELoss
from prototype.data import (
    build_imagenet_train_dataloader, build_imagenet_test_dataloader,
    build_imagenet_search_dataloader, build_custom_dataloader
)

try:
    from SpringCommonInterface import SpringCommonInterface
except ImportError:
    SpringCommonInterface = object


class PrototypeHelper(SpringCommonInterface):
    """Unitied interface for spring"""

    external_model_builder = {}

    def __init__(self, config, metric_dict=None, work_dir=None, ckpt_dict=None):
        """
        Args:
            config (dict): All the configs to build the task
            metric_dict (dict): Dict of Prometheus logger system's instances. Currently it contains two keywords:
                         - process: Gauge type. 0 - 100. Indicates the process of training.
                         - eta: Gauge type. hh:mm:ss. Indicates the estimated remaining time.
            work_dir (str): Task's root folder. Please save all the intermediate results in work_dir.
            ckpt_dict (dict): It contains all the saved k-v pairs for resuming. Only resume task when it's not None
        """
        super(PrototypeHelper, self).__init__(config, metric_dict, work_dir, ckpt_dict)
        # configuration for model training, evaluating, etc.
        self.config = config
        self.config_copy = copy.deepcopy(config)
        self.metric_dict = metric_dict
        self.work_dir = work_dir
        self.ckpt_dict = ckpt_dict

        self._setup_env()
        self._resume(ckpt_dict)
        self._build()
        self._pre_train()
        self.end_time = time.time()

    def _setup_env(self):
        # distribution information
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        # directories
        self.path = EasyDict()
        self.path.root_path = self.work_dir
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # create tensorboard logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # create logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # others
        torch.backends.cudnn.benchmark = True

    def _resume(self, ckpt_dict=None):
        '''
        The ckpt_dict owns higher priority than element's resuming
        '''
        if ckpt_dict:
            self.state = ckpt_dict
            self.curr_step = self.state['last_iter']
            self.logger.info(f"Recovering from ckpt_dict, keys={list(self.state.keys())}")
        else:
            # load pretrain
            if hasattr(self.config.saver, 'pretrain'):
                self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
                self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
                if hasattr(self.config.saver.pretrain, 'ignore'):
                    self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
                self.curr_step = self.state['last_iter']
            else:
                self.state = {'last_iter': 0}
                self.curr_step = 0

    def _build(self):
        self.build_model()
        self._build_optimizer()
        self._build_data()
        self._build_lr_scheduler()
        # load pretrain state to model
        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    # sci
    def build_model(self):
        '''
        Perform the model building operation only. ``Do not include loading weights``.

        The interface must be used in the model building process by init function.
        Don't implement like below, InterfaceImplement is your implementation of interface class:

        Returns:
            torch.nn.Module: The Pytorch GPU model.
        '''
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.model.cuda()
        # count flops and params
        count_params(self.model)
        count_flops(self.model, input_shape=[1, 3, self.config.data.input_size, self.config.data.input_size])
        # handle fp16
        if self.config.optimizer.type in ['FP16SGD', 'FusedFP16SGD', 'FP16RMSprop']:
            self.fp16 = True
        else:
            self.fp16 = False

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if self.config.optimizer.get('fp16_normal_bn', False):
                self.logger.info('using normal bn for fp16')
                link.fp16.register_float_module(
                    link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(
                    torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.info('using normal fc for fp16')
                link.fp16.register_float_module(
                    torch.nn.Linear, cast_args=True)
            link.fp16.init()
            self.model.half()

        self.model = DistModule(self.model, self.config.dist.sync)
        return self.model

    def _build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        # divide param_groups
        pconfig = {}
        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}
        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])
        param_group, type2num = param_group_all(self.model, pconfig)
        opt_config.kwargs.params = param_group
        self.optimizer = optim_entry(opt_config)
        # load optimizer
        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])
        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
            # load EMA
            if 'ema' in self.state:
                self.ema.load_state_model(self.state['ema'])
        else:
            self.ema = None

    def _build_lr_scheduler(self):
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer.optimizer if isinstance(self.optimizer, FP16SGD) or \
            isinstance(self.optimizer, FP16RMSprop) else self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def _build_data(self):
        self.config.data.last_iter = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch

        self.data_loaders = {}
        key_list = list(self.config.data.keys())
        for data_type in key_list:
            if data_type in ['train', 'test', 'val', 'arch', 'inference']:
                if self.config.data.type == 'imagenet':
                    # imagenet type
                    if data_type == 'train':
                        loader = build_imagenet_train_dataloader(self.config.data)
                    elif data_type == 'test':
                        loader = build_imagenet_test_dataloader(self.config.data)
                    else:
                        loader = build_imagenet_search_dataloader(self.config.data)
                else:
                    # custom type
                    loader = build_custom_dataloader(data_type, self.config.data)
                self.data_loaders.update({data_type: loader['loader']})

        if self.data_loaders['train'] is not None:
            self.total_step = len(self.data_loaders['train'])
        else:
            self.total_step = 0

    def _pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        label_smooth = self.config.get('label_smooth', 0.0)
        self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        if label_smooth > 0:
            self.logger.info('using label_smooth: {}'.format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.mixup = self.config.get('mixup', 1.0)
        self.cutmix = self.config.get('cutmix', 0.0)
        if self.mixup < 1.0:
            self.logger.info('using mixup with alpha of: {}'.format(self.mixup))
        if self.cutmix > 0.0:
            self.logger.info('using cutmix with alpha of: {}'.format(self.cutmix))

    # sci
    def get_model(self):
        '''
        Return the Pytorch GPU Model.
        The model should not be built every time this interface is called.

        Returns:
            torch.nn.Module: The Pytorch GPU model to train/mimic/prune/quant/...
        '''
        return self.model

    # sci
    def get_optimizer(self):
        '''
        Return the optimizer of the Pytorch GPU Model.
        The optimizer should not be built every time this interface is called.

        Returns:
            torch.optim.Optimizer: The optimizer of the Pytorch GPU model
        '''
        return self.optimizer

    # sci
    def get_scheduler(self):
        '''
        It should not be built every time this interface is called.

        Returns:
            torch.optim.lr_scheduler: The scheduler
        '''
        return self.lr_scheduler

    # sci
    def get_dummy_input(self):
        '''
        Request the input for forwarding the model.
        It will be used to calc FLOPs and so on.
        Make sure the returned input can support operation: model(get_dummy_input())

        Returns:
            object: The dummy input for current task.
        '''
        input = torch.zeros(1, 3, self.config.data.input_size, self.config.data.input_size)
        input = input.cuda().half() if self.fp16 else input.cuda()
        return input

    # sci
    def get_dump_dict(self):
        '''
        Returns:
            dict: Custom dict to be dumped by torch.save
        '''
        dict_to_dump = {}
        dict_to_dump['config'] = self.config_copy
        dict_to_dump['model'] = self.model.state_dict()
        dict_to_dump['optimizer'] = self.optimizer.state_dict()
        dict_to_dump['last_iter'] = self.curr_step
        if self.ema is not None:
            dict_to_dump['ema'] = self.ema.state_dict()
        return dict_to_dump

    # sci
    def get_batch(self, batch_type='train'):
        '''
        Return the batch of the given batch_type. The valid batch_type is set in config.
        e.g. Your config file is like below, then the interface should return corresponding batch
        when batch_type is train, test or val.

        The returned batch will be used to call `forward` function of SpringCommonInterface.
        The first item will be used to forward model like: model(get_batch('train')[0]).
        Please make sure the first item in the returned batch be FP32 and in GPU.
        If your model is FP16, please do convert in your model's forward function.

        Args:
            batch_type (str): Default: 'train'. It can also be 'val', 'test' or other custom type.

        Returns:
            tuple: a tuple of batch (input, label)
        '''
        assert batch_type in self.data_loaders
        if not hasattr(self, 'data_iterators'):
            self.data_iterators = {}
        if batch_type not in self.data_iterators:
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
        else:
            iterator = self.data_iterators[batch_type]

        try:
            batch = next(iterator)
        except StopIteration as e:  # noqa
            iterator = self.data_iterators[batch_type] = iter(self.data_loaders[batch_type])
            batch = next(iterator)
        return batch['image'], batch['label']

    # sci
    def get_total_iter(self):
        '''
        Return the total iteration of the Task.
        Note that even the task is resumed, the returned value should not be changed.

        Returns:
            total_iter (int): the total iteration of the Task. Please convert epoch to iter if needed
        '''
        return self.config.data.max_iter

    # sci
    @staticmethod
    def load_weights(model, ckpt_dict):
        '''
        Static Function. Load weights for model from ckpt

        Args:
            model (torch.nn.Module): Pytorch GPU model to be loaded
            ckpt_dict (dict): checkpoint dict to resume task
        '''
        model.load_state_dict(ckpt_dict['model'], strict=True)

    # sci
    def forward(self, batch):
        '''
        Forward with the given batch and return the loss, e.g. the batch from `get_batch` interface.
        Do not manually change the magnitude of loss, like dividing the world size.
        Do not manually change the model's state, i.e. model.train() or model.eval()

        Args:
            batch (tuple): the batch for forwarding the model

        Returns:
            loss (torch.cuda.Tensor): loss tensor in GPU, the loss of the given batch and current model.
        '''
        # measure data loading time
        self.meters.data_time.update(time.time() - self.end_time)
        input, target = batch[0], batch[1]
        input = input.cuda().half() if self.fp16 else input.cuda()
        target = target.squeeze().view(-1).cuda().long()
        # forward
        logits = self.model(input)
        loss = self.criterion(logits, target)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, self.topk))

        reduced_loss = loss.clone() / self.dist.world_size
        reduced_prec1 = prec1.clone() / self.dist.world_size
        reduced_prec5 = prec5.clone() / self.dist.world_size

        self.meters.losses.reduce_update(reduced_loss)
        self.meters.top1.reduce_update(reduced_prec1)
        self.meters.top5.reduce_update(reduced_prec5)
        return loss

    # sci
    def backward(self, loss):
        '''
        Backward with the given loss.
        Please don't modify the magnitude of loss (divid by the world size or multiply loss weight).
        The gradient should be synchronized by all_reduce with sum operation.
        Do not manually change the model's state, i.e. model.train() or model.eval()

        Args:
            loss (torch.cuda.Tensor): loss tensor in GPU
        '''
        self.optimizer.zero_grad()
        if self.fp16:
            self.optimizer.backward(loss)
            self.model.sync_gradients()
        else:
            loss.backward()
            self.model.sync_gradients()

    # sci
    def update(self):
        '''
        Update the model with current calculated gradients.
        The scheduler should also be stepped.
        '''
        self.curr_step += 1    # move from forward() to update()
        self.lr_scheduler.step(self.curr_step)
        self.optimizer.step()
        # EMA
        if self.ema is not None:
            self.ema.step(self.model, curr_step=self.curr_step)

        # set metric_dict
        if self.metric_dict is not None and 'eta' in self.metric_dict:
            self.metric_dict['eta'].set(self.get_eta())
        if self.metric_dict is not None and 'progress' in self.metric_dict:
            self.metric_dict['progress'].set(self.get_progress())
        self.meters.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()

    def get_eta(self):
        return (self.total_step - self.curr_step) * self.meters.batch_time.avg

    def get_progress(self):
        return self.curr_step / self.total_step * 100

    # sci
    def train(self):
        '''
        Perform the entire training process
        '''
        for i in range(self.total_step):
            batch = self.get_batch()
            loss = self.forward(batch)
            self.backward(loss / self.dist.world_size)
            self.update()

            # measure elapsed time
            self.meters.batch_time.update(time.time() - self.end_time)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            curr_step = self.curr_step
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (self.total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                log_msg = f'Iter: [{curr_step}/{self.total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})' \

                self.logger.info(log_msg)

            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                metrics = self.evaluate()
                if self.ema is not None:
                    self.ema.load_ema(self.model)
                    ema_metrics = self.evaluate()
                    self.ema.recover(self.model)
                    if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                        self.tb_logger.add_scalars('acc1_val', {'ema': ema_metrics.metric['top1']}, curr_step)
                        self.tb_logger.add_scalars('acc5_val', {'ema': ema_metrics.metric['top5']}, curr_step)

                # testing logger
                if self.dist.rank == 0 and self.config.data.test.evaluator.type == 'imagenet':
                    self.tb_logger.add_scalar('acc1_val', metrics.metric['top1'], curr_step)
                    self.tb_logger.add_scalar('acc5_val', metrics.metric['top5'], curr_step)

                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    if self.ema is not None:
                        self.state['ema'] = self.ema.state_dict()
                    torch.save(self.state, ckpt_name)

            self.end_time = time.time()

    # sci
    @torch.no_grad()
    def evaluate(self):
        '''
        Do evaluation and return Metric Class instance

        Returns:
            Metric: Metric class instance
        '''
        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.data_loaders['test']):
            input = batch['image']
            label = batch['label']
            input = input.cuda().half() if self.fp16 else input.cuda()
            label = label.squeeze().view(-1).cuda().long()
            # compute output
            logits = self.model(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            self.data_loaders['test'].dataset.dump(writer, batch)

        writer.close()
        link.barrier()
        if self.dist.rank == 0:
            metrics = self.data_loaders['test'].dataset.evaluate(res_file)
            self.logger.info(json.dumps(metrics.metric, indent=2))
        else:
            metrics = {}
        link.barrier()
        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        self.model.train()
        return metrics

    # sci
    @staticmethod
    def build_model_helper(config_dict=None):
        '''
        Static function. Build a model from the given config_dict.

        Args:
            config (dict): the config that contains model information

        Returns:
            torch.nn.Module: The Pytorch GPU model.
        '''
        if not isinstance(config_dict, EasyDict):
            config_dict = EasyDict(config_dict)
        model = model_entry(config_dict.model)
        model.cuda()

        if config_dict.optimizer.type in ['FP16SGD', 'FusedFP16SGD', 'FP16RMSprop']:
            fp16 = True
        else:
            fp16 = False

        if fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if config_dict.optimizer.get('fp16_normal_bn', False):
                link.fp16.register_float_module(link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            if config_dict.optimizer.get('fp16_normal_fc', False):
                link.fp16.register_float_module(torch.nn.Linear, cast_args=True)
            link.fp16.init()
            model.half()

        model = DistModule(model, config_dict.dist.sync)
        return model

    # sci
    def show_log(self):
        '''
        Display the log of current iteration. The interface will be called after
        forward/backward/update
        '''
        curr_step = self.curr_step
        current_lr = self.lr_scheduler.get_lr()[0]
        remain_secs = (self.total_step - curr_step) * self.meters.batch_time.avg
        remain_time = datetime.timedelta(seconds=round(remain_secs))
        finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
        log_msg = f'Iter: [{curr_step}/{self.total_step}]\t' \
            f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
            f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
            f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
            f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
            f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
            f'LR {current_lr:.4f}\t' \
            f'Remaining Time {remain_time} ({finish_time})' \

        self.logger.info(log_msg)

    # sci
    @classmethod
    def add_external_model(cls, name, callable_object):
        '''
        Add external model into the element. After this interface is called, the element should
        be able to build model by the given ``name``.

        Args:
            name (str): The identifier of callable_object
            callable_object (callable object): A class or a function that is callable to build a torch.nn.Module model.
        '''
        cls.external_model_builder[name] = callable_object

    # sci
    def convert_model(self, type="skme"):
        '''
        Dump pytorch model to deployable type model (skme or caffe).
        More about SKME（AKA onnx with oplib): https://confluence.sensetime.com/pages/viewpage.action?pageId=135889068
        Recommand spring.nart.tools.pytorch.convert_caffe_by_return for caffe,
        spring.nart.tools.pytorch.convert_onnx_by_return for skme

        Args:
            type: deploy model type. "skme" or "caffe"

        Returns:
            dict: A dict of model file (string) and type.
            The format should be {"model": [model1, model2, ...], "type": "skme"}.
        '''
        if type == 'skme':
            self.logger.warning('skme is not supported yet, we support convert to caffemodel for now')
        return {
            'model': [self.to_caffe()],
            'type': type
        }

    # sci
    def get_kestrel_parameter(self):
        '''
        get kestrel plugin parameter file.
        parameters contains model_files and other options (which bind to relate kestrel plugin)
        model_files must follow this contract:

        Returns:
            str: Kestrel plugin parameter.json content.
        '''
        assert hasattr(self.config, 'to_kestrel')
        kestrel_config = self.config.to_kestrel
        kestrel_param = EasyDict()
        # default: ImageNet statistics
        kestrel_param['pixel_means'] = kestrel_config.get('pixel_means', [123.675, 116.28, 103.53])
        kestrel_param['pixel_stds'] = kestrel_config.get('pixel_stds', [58.395, 57.12, 57.375])
        # default: True/True/UNKNOWN
        kestrel_param['is_rgb'] = kestrel_config.get('is_rgb', True)
        kestrel_param['save_all_label'] = kestrel_config.get('save_all_label', True)
        kestrel_param['type'] = kestrel_config.get('type', 'UNKNOWN')
        # class label
        if hasattr(kestrel_config, 'class_label'):
            kestrel_param['class_label'] = kestrel_config['class_label']
        else:
            # default: imagenet
            kestrel_param['class_label'] = {}
            kestrel_param['class_label']['imagenet'] = {}
            kestrel_param['class_label']['imagenet']['calculator'] = 'bypass'
            kestrel_param['class_label']['imagenet']['labels'] = [str(i) for i in np.arange(self.num_classes)]
            kestrel_param['class_label']['imagenet']['feature_start'] = 0
            kestrel_param['class_label']['imagenet']['feature_end'] = 0

        return json.dumps(kestrel_param)

    # sci
    def get_epoch_iter(self, loader_name):
        '''
        Return the epoch iteration of the loader with the given loader_name.
        The returned value equals to one epoch iteration, i.e. len(self.xxx_loader)
        This interface is different with `get_total_iter`, which is required to return
        the total iteration of the training process.

        Args:
            loader_name (str): loader's name.

        Returns:
            int: The epoch iteration of the loader with the given loader_name.
        '''
        return len(self.data_loaders[loader_name])

    # sensespring
    def to_caffe(self, save_prefix='model', input_size=None):
        from spring.nart.tools import pytorch

        with pytorch.convert_mode():
            pytorch.convert(self.model.float(),
                            [(3, self.config.data.input_size,
                              self.config.data.input_size)],
                            filename=save_prefix,
                            input_names=['data'],
                            output_names=['out'])

    # sensespring
    def to_kestrel(self, save_to=None):
        assert hasattr(self.config, 'to_kestrel')
        prefix = 'model'
        self.logger.info('Converting Model to Caffe...')
        if self.dist.rank == 0:
            self.to_caffe(save_prefix=prefix)
        link.synchronize()
        self.logger.info('To Caffe Done!')

        prototxt = '{}.prototxt'.format(prefix)
        caffemodel = '{}.caffemodel'.format(prefix)
        # default version '1.0.0'
        # acquire version and model_name
        version = self.config.to_kestrel.get('version', '1.0.0')
        model_name = self.config.to_kestrel.get('model_name', self.config.model.type)
        kestrel_model = '{}_{}.tar'.format(model_name, version)
        to_kestrel_yml = 'temp_to_kestrel.yml'
        # acquire to_kestrel params
        kestrel_param = json.loads(self.get_kestrel_parameter())
        with open(to_kestrel_yml, 'w') as f:
            yaml.dump(kestrel_param, f)

        cmd = 'python -m spring.nart.tools.kestrel.classifier {} {} -v {} -c {} -n {}'.format(
            prototxt, caffemodel, version, to_kestrel_yml, model_name)
        self.logger.info('Converting Model to Kestrel...')
        if self.dist.rank == 0:
            os.system(cmd)
        link.synchronize()
        self.logger.info('To Kestrel Done!')
        if save_to is None:
            save_to = self.config['to_kestrel']['save_to']
        shutil.move(kestrel_model, save_to)
        self.logger.info('save kestrel model to: {}'.format(save_to))

        # convert model to nnie
        nnie_cfg = self.config.to_kestrel.get('nnie', None)
        if nnie_cfg is not None:
            nnie_model = 'nnie_{}_{}.tar'.format(model_name, version)
            nnie_cfg_path = generate_nnie_config(nnie_cfg, self.config)
            nnie_cmd = 'python -m spring.nart.switch -c {} -t nnie {} {}'.format(
                nnie_cfg_path, prototxt, caffemodel)
            self.logger.info('Converting Model to NNIE...')
            if self.dist.rank == 0:
                os.system(nnie_cmd)
                # refactor json
                assert os.path.exists("parameters.json")
                with open("parameters.json", "r") as f:
                    params = json.load(f)
                params["model_files"]["net"]["net"] = "engine.bin"
                params["model_files"]["net"]["backend"] = "kestrel_nart"
                with open("parameters.json", "w") as f:
                    json.dump(params, f, indent=2)
                tar_cmd = 'tar cvf {} engine.bin engine.bin.json meta.json meta.conf \
                    parameters.json category_param.json'.format(nnie_model)
                os.system(tar_cmd)
                self.logger.info(f"generate {nnie_model} done!")
            shutil.move(nnie_model, save_to)
            link.synchronize()
            self.logger.info('To NNIE Done!')

        return save_to

    # sci
    @torch.no_grad()
    def inference(self):
        '''
        Inference the inference dataset and save the raw results (not the evaluation value) in the result file.
        The inference dataset and correspoding config should be set at the config file.
        The result file should be in json format.

        Returns:
            str: The absolute path to the saved results file.
        '''
        assert 'inference' in self.data_loaders.keys()
        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'infer_results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.data_loaders['inference']):
            input = batch['image']
            input = input.cuda().half() if self.fp16 else input.cuda()
            # compute output
            logits = self.model(input)
            scores = F.softmax(logits, dim=1)
            # compute prediction
            _, preds = logits.data.topk(k=1, dim=1)
            preds = preds.view(-1)
            # update batch information
            batch.update({'prediction': preds})
            batch.update({'score': scores})
            # save prediction information
            self.data_loaders['inference'].dataset.dump(writer, batch)

        writer.close()
        link.barrier()
        if self.dist.rank == 0:
            infer_res_file = self.data_loaders['inference'].dataset.inference(res_file)
        else:
            infer_res_file = None
        link.barrier()
        # broadcast file to other process
        infer_res_file = broadcast_object(infer_res_file)
        self.model.train()
        return infer_res_file
