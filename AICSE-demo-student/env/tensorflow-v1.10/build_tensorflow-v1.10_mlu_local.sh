#!/bin/bash

# This is for cambricon internal use.
# This build script is meant to build tensorflow on local disk and
# output the virtual env to NAS so that you could debug on other MLU machines.
# Usage:
# 1. Pull tensorflow code on /local/yourname.
# 2. Build with this script
# 3. You'll find tensorflow output directory on /proj/framework/yourname/tfvenv_mlu
# 4. source tfvenv_mlu/bin/activate
# 5. pip install -U protobuf for the first time

set -e

USER_PROJ_PATH=$(find /projs/ -maxdepth 2 -type d -name `whoami`)
[ -z "$USER_PROJ_PATH" ] && {
    echo "Can't find user " $(whoami) " on /projs"
    exit 1
}

NAS_VENV_PATH=${USER_PROJ_PATH}/tfvenv_mlu

export PATH_TFVENV_MLU=${NAS_VENV_PATH}
echo "PATH_TFVENV_MLU=${PATH_TFVENV_MLU}"

./build_tensorflow-v1.10_mlu.sh
