#!/usr/bin/env bash
# Install the CUDA compiler and static runtime inside the manylinux image.
set -euo pipefail

prefix="${1:?usage: $0 PREFIX}"
rhel="rhel$(rpm -E '%{rhel}')"
repo="https://developer.download.nvidia.com/compute/cuda/repos/$rhel/x86_64"
curl -fsSL --retry 3 "$repo/cuda-$rhel.repo" -o /etc/yum.repos.d/cuda.repo
dnf install -y --setopt=install_weak_deps=False cuda-nvcc-13-5 cuda-cudart-devel-13-5
install -Dm644 /usr/local/cuda-13.5/EULA.txt "$prefix/licenses/cuda/EULA.txt"
