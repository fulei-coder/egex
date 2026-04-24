#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

find $SCRIPT_DIR -type f -exec chmod +x {} \;

echo "[INFO] $SCRIPT_DIR 内所有文件已添加可执行权限。"
