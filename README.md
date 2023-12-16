# Parallel Computing Final Project

## Tasks

### generate
Generate build files.
```
cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=RelWwithDebInfo -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx .
```

### build
Build project.
```
ninja -C build
```
