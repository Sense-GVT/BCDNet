PYTHONPATH=$PYTHONPATH:../../../ GLOG_vmodule=MemcachedClient=-1 \
spring.submit run -r --mpi=pmi2 -n32 --gpu --gpu-type=1080ti --cpus-per-task=5 \
"python -u -m prototype.solver.cls_solver_dist --config config.yaml" 
