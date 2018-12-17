# drake-pytorch
Combining pytorch (python and c++ bindings) with drake for neural network systems.
A big rip-off of drake-shambala drake\_cmake\_installed
https://github.com/RobotLocomotion/drake-shambhala/tree/master/drake\_cmake\_installed
nicely supports multiple srcs for little experiments

# to run
```
#rm -rf build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="/opt/drake;`pwd`/../../torchlib_cpu/" ../
make
#make test
```

