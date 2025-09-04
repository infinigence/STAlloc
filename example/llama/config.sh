#!/bin/bash
# Common configuration file for LLaMA experiments
# This file contains environment settings and imports paths configuration

# Environment settings
export PYTHONPATH=/workspace/Megatron-LM-080 # path to Megatron-LM core_r0.8.0

# Import dataset and tokenizer paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
