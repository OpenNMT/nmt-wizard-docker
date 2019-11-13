#! /bin/sh
# This script helps factorizing the installation of Python packages in Dockerfiles.

set -e

# Install system packages to build Python extensions.
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    python-dev \
    wget
apt-get clean
rm -rf /var/lib/apt/lists/*

# Install pip.
wget -nv https://bootstrap.pypa.io/get-pip.py
python get-pip.py
rm get-pip.py

# Forward pip arguments.
pip --no-cache-dir install $@

# Remove build specific system packages.
apt-get autoremove -y --purge \
    build-essential \
    ca-certificates \
    wget
