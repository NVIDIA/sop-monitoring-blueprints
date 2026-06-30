---
name: nvidia-driver-install
description: >-
  Install NVIDIA driver and CUDA 13.0 toolkit when nvidia-smi is not detected.
  Use when GPU detection fails, after a fresh OS install, or when the driver
  needs to be upgraded to 580.x for VSS SOP.
---

# NVIDIA Driver & CUDA 13.0 Installation

## When to Use

- `nvidia-smi` command fails or is not found
- Driver version is below `580.x` and needs upgrading
- Fresh machine setup before deploying VSS SOP

## Requirements

| Item      | Value                          |
|-----------|--------------------------------|
| OS        | Ubuntu 22.04 / 24.04 (x86_64) |
| CUDA      | 13.0                           |
| Driver    | 580.65.06 (22.04) / 580.105.08 (24.04) |
| Hardware  | H100 / H200 / A100 / RTX PRO 6000 Blackwell |

---

## Step 1 — Detect current state

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>&1
```

If the command succeeds and shows driver `580.x`, skip to **Step 4 (Verify)**.
If it fails or shows an older driver, continue with Step 2.

## Step 2 — Remove conflicting drivers (if any)

Only run this if upgrading from an older driver or a previous install is broken:

```bash
sudo apt-get purge -y 'nvidia-*' 'libnvidia-*' 2>/dev/null
sudo apt-get autoremove -y
sudo rm -f /etc/apt/preferences.d/cuda-repository-pin-600
```

## Step 3 — Install CUDA 13.0 + Driver 580

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
sudo apt-get install -y cuda-drivers

rm -fv cuda-repo-ubuntu2204-13-0-local_13.0.0-580.65.06-1_amd64.deb
```

After installation completes, load the driver **without rebooting**:

```bash
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
```

If `modprobe` fails (e.g. the old driver is still loaded), unload the existing modules first:

```bash
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset
sudo modprobe nvidia_drm
```

If `rmmod` fails because modules are in use, a reboot is the only option:

```bash
sudo reboot
```

## Step 4 — Verify

Confirm the driver and CUDA are working:

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
```

Expected output should show driver version `580.65.06` (Ubuntu 22.04) or `580.105.08` (Ubuntu 24.04) and the correct GPU(s).

Also verify CUDA toolkit:

```bash
nvcc --version 2>/dev/null || /usr/local/cuda-13.0/bin/nvcc --version
```

If `nvcc` is not found in `PATH`, add CUDA to the shell profile:

```bash
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 5 — Post-install: NVIDIA Container Toolkit

After the driver is installed, the NVIDIA Container Toolkit is needed for Docker GPU access. See [`prerequisites.md`](prerequisites.md) section 3 for full instructions, or run:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify Docker can see the GPU:

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi 2>&1 | head -8
```

---

## Troubleshooting

- **`nvidia-smi` shows `NVIDIA-SMI has failed`** — driver installed but not loaded. Try `sudo modprobe nvidia`. If that fails, reboot.
- **`dpkg` lock error** — another apt process is running. Wait or kill it: `sudo kill -9 $(lsof -t /var/lib/dpkg/lock-frontend)`.
- **Secure Boot blocks the driver** — disable Secure Boot in BIOS/UEFI, or enroll the NVIDIA signing key via `sudo mokutil`.
- **Wrong driver version after install** — ensure no conflicting PPA or distro driver. Run Step 2 to purge, then reinstall.

---

## License

Use of this skill is governed by the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode.en) and the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
