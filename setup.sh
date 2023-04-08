#!/usr/bin/env bash

# source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh
# source ~/hailing.env

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PATH="${SCRIPT_DIR}:${SCRIPT_DIR}/bin:${PATH}"
