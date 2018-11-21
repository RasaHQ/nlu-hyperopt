#!/bin/bash
set -Eeuo pipefail

#######################################
#
# installing ansible and roles
#
#######################################
echo "Installing pip and ansible"
sudo apt-get install python
curl -O https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install ansible

echo "Installing docker role"
ansible-galaxy install angstwad.docker_ubuntu
echo "Docker installed"

#######################################
#
# running the ansible playbook
# (starts docker und serves platform)
#
#######################################
echo "Running playbook"
ansible-playbook -i "localhost," -c local playbook.yml
