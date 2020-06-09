# GluonCV lite models
Deploy gluon-cv models using TVM

## Build instruction
Always use `git clone --recursive`, if not, we can update tvm submodule `git submodule update --recursive --init`.

```
mkdir -p build && cd build
cmake ..
make
```

TODO(): build libjpeg and libpng statically and link to them so it should work on user's clean environment.
