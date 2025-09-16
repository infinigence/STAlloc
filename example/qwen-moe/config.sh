#!/bin/bash
# Common configuration file for LLaMA experiments
# This file contains environment settings and imports paths configuration

# Environment settings
export PYTHONPATH=/path/to/Megatron-LM-010 # path to Megatron-LM core_r0.10.0

# Import dataset and tokenizer paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../paths.sh"
