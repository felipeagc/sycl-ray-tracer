# SYCL Ray Tracer

Implementation of a GPU-based ray tracer using SYCL and Embree using both megakernel and wavefront approaches.

![sponza](https://github.com/felipeagc/sycl-ray-tracer/assets/17355488/22a49ad9-f63f-48ba-84ab-fa4d2b7e595e)

## Intel oneAPI install on Debian

### Install drivers
```bash
sudo apt update
sudo apt install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy/production/2328 unified" | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt update
sudo apt install -y linux-headers-$(uname -r) flex bison xpu-smi
sudo apt install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo
sudo apt install -y \
  libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev
```

### Add user to render group

```bash
sudo gpasswd -a ${USER} render
```

### Install oneAPI toolkits

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install intel-basekit
sudo apt install intel-renderkit
sudo apt install intel-oneapi-runtime-libs
```

## Authoring scenes in Blender

Export the scene as GLTF.

### Sky color
Add a custom property to the scene called `sky_color` with the type float array with 3 elements.

### Materials

#### Dielectric material
In the Principled BSDF node, set the IOR to something other than the default (1.5) and the transmission weight to a non-zero value.

#### Metallic material
Used if the metallic property in the Principled BSDF node is non-zero.

The metallic value is not actually used in lighting calculations,
it just identifies that the material type should be metallic.
The roughness value is used for shading, though.

#### Diffuse material
Used if no other criteria matches.

## Tasks

### generate
Generate build files.
```
rm -rf build
cmake -Bbuild -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx .
```

### generate-debug
Generate build files.
```
rm -rf build
cmake -Bbuild -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx .
```

### build
Build project.
```
ninja -C build
```

### watch
Build project.
```
watchexec --no-vcs-ignore -e cpp,hpp,h,glb,gltf,txt "ninja -C build && ./build/raytracer"
```
