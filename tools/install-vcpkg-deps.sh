#!/usr/bin/env bash
# Install static Boost.Serialization and GSL with vcpkg for PyPI wheels.
#
# Usage: tools/install-vcpkg-deps.sh PREFIX TRIPLET
#
# Checks out vcpkg at the builtin-baseline commit of vcpkg.json, which pins
# every library version. Installs into PREFIX/installed and copies license
# texts to PREFIX/licenses. cibuildwheel runs this as `before-all` on Linux
# and macOS; Windows uses install-vcpkg-deps.ps1.
set -euo pipefail

PREFIX="${1:?usage: $0 PREFIX TRIPLET}"
TRIPLET="${2:?usage: $0 PREFIX TRIPLET}"
project="$(cd "$(dirname "$0")/.." && pwd)"
baseline="$(sed -n 's/.*"builtin-baseline": *"\([0-9a-f]*\)".*/\1/p' "$project/vcpkg.json")"
vcpkg_root="$PREFIX/vcpkg"

if [[ "$(uname -s)" == Linux ]]; then
    # manylinux images lack zip, which vcpkg uses for its binary cache.
    command -v zip >/dev/null || dnf install -y zip
fi

if [[ ! -x "$vcpkg_root/vcpkg" ]]; then
    git init -q "$vcpkg_root"
    git -C "$vcpkg_root" fetch -q --depth 1 \
        https://github.com/microsoft/vcpkg.git "$baseline"
    git -C "$vcpkg_root" -c advice.detachedHead=false checkout -q FETCH_HEAD
    "$vcpkg_root/bootstrap-vcpkg.sh" -disableMetrics
fi

"$vcpkg_root/vcpkg" install \
    --x-manifest-root="$project" \
    --x-install-root="$PREFIX/installed" \
    --overlay-triplets="$project/tools/vcpkg-triplets" \
    --triplet="$TRIPLET"

share="$PREFIX/installed/$TRIPLET/share"
mkdir -p "$PREFIX/licenses/boost" "$PREFIX/licenses/gsl"
cp "$share/boost-serialization/copyright" "$PREFIX/licenses/boost/LICENSE_1_0.txt"
cp "$share/gsl/copyright" "$PREFIX/licenses/gsl/COPYING"
