# Parallel Computing Final Project

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
