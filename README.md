# Artifact for "STAlloc: Enhancing Memory Efficiency in Large-Scale Model Training with Spatio-Temporal Planning"

This repository contains the artifact accompanying our paper:

> Zixiao Huang, Junhao Hu, Hao Lin, Chunyang Zhu, Yueran Tang, Quanlu Zhang, Zhen Guo, Zhenhua Li, Shengen Yan, Zhenhua Zhu, Guohao Dai, and Yu Wang.  
> **STAlloc: Enhancing Memory Efficiency in Large-Scale Model Training with Spatio-Temporal Planning**  
> To appear in *Proceedings of the 21st ACM European Conference on Computer Systems (EuroSys '26)*.  

## Description
* STAlloc is a memory tool used to reduce memory fragmentation and improve memory utilization.
* STAlloc actively leverages the predictable memory allocation pattern of large model training to perform ahead-of-time memory allocation planning.

## Compilation
* `cd Allocator && make`
* See `Allocator/README.md` for more details

## Env
| Env | Value | Description |
| --- | ----- | ----------- |
|`STALLOC_MODE` |`[Torch (default), Trace, Alloc]`| set the mode of STAlloc.
|`STALLOC_LIB_PATH` | path to STAlloc lib, required when using `STALLOC_MODE` is `Trace` or `Alloc`.
|`STALLOC_MODEL_INFO_PATH` | path to save model memory info, required when `STALLOC_MODE` is `Trace` or `Alloc`.
|`STALLOC_DYNAMIC` | `[0 (default), 1]`| required in MoE model(without batchgemm) when `STALLOC_MODE` is `Trace` or `Alloc`.
|`STALLOC_LOG_LEVEL` |`[0, 1, 2, 3 (default)]`| set the log level of STAlloc, the smaller the value, the more detailed the output content.
|`STALLOC_STATIC_FALLBACK` |`[0 (default), 1]`| enable fallback in static-alloc, which will affect the performance
|`STALLOC_TRACE_FAST_MODE` |`[0 (default), 1]`| use faster dynamic-allocator to trace, but this may lead to OOM, when the memory required by the model has reached the limit of GPU.


## Preparation
### pretrain_xxx.py
```python
"""
At the beginning of pretrain.py, import STAlloc
"""
from STAlloc.utils.hook_model import hook_memory_model

def model_provider(...):
    ...
    #return model 
    return hook_memory_model(model, args)
```

### train script
Add the following code to train script.
```shell
export STALLOC_MODE=Trace
export STALLOC_TRACE_FAST_MODE=1  # may lead to OOM
STALLOC_PATH=YourPath
export STALLOC_LIB_PATH=${STALLOC_PATH}/Allocator

MODEL_TAG=llama3-70b-tp8pp8mbs1gbs128-node${RANK}
MEMORY_SAVED_DIR=/workspace/allocator_case
export STALLOC_MODEL_INFO_PATH=${MEMORY_SAVED_DIR}/${MODEL_TAG}
if [ "$STALLOC_MODE" == "Trace" ]; then
    if [ -e "${STALLOC_MODEL_INFO_PATH}/trace" ]; then
       rm -rf ${STALLOC_MODEL_INFO_PATH}/trace
    fi
    mkdir -p ${STALLOC_MODEL_INFO_PATH}/trace
    mkdir -p ${STALLOC_MODEL_INFO_PATH}/output
elif [ "$STALLOC_MODE" == "Alloc" ]; then
    export STALLOC_LOG_LEVEL=1
    if [ ! -e "${STALLOC_MODEL_INFO_PATH}/output/plan" ]; then
       exit 1
    fi
fi

# !!! If you set "STALLOC_MODE=Trace", please make sure that "train-iter=3" and "eval-iter=1".
```

## Usage

* After preparation work is done, run the memory tools by following steps.

### Step-1
* Set `export STALLOC_MODE=Torch` and run the training script.
* Check if the GPU memory fragmentation of torch is severe.
    * Find this form of line in log : `dev0 : max_reserved:xx.xx, max_allocated:xx.xx, utilization:xx.xx%`
    * The smaller the `utilization`, the more severe of the fragmented storage.
    * It is generally considered that the fragmentation problem is severe when `utilization` is less than 90%.
* OOM in Torch Mode does not mean that Trace will also OOM.

### Step-2
* Set `export STALLOC_MODE=Trace` and run the training script.

### Step-3
* Generate Plan. run the following command
* `cd ${STALLOC_PATH}/Synthesizer`
* `python main.py  -model-memory-dir=XXXX`
* exec `python mian.py --help` for more details

### Step-4
* Set `export STALLOC_MODE=Alloc` and run the training script.

## Artifact Experiments
See `example`.

## Others
If you are using other version of Megatron-LM, you may need to check the path of patch functions, and modify in `utils/memory_patcher.py`.

## License
This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{huang2026stalloc,
  title     = {STAlloc: Enhancing Memory Efficiency in Large-Scale Model Training with Spatio-Temporal Planning},
  author    = {Zixiao Huang and Junhao Hu and Hao Lin and Chunyang Zhu and Yueran Tang and Quanlu Zhang and Zhen Guo and Zhenhua Li and Shengen Yan and Zhenhua Zhu and Guohao Dai and Yu Wang},
  booktitle = {Proceedings of the 21st ACM European Conference on Computer Systems (EuroSys '26)},
  year      = {2026},
  publisher = {ACM},
  note      = {To appear in EuroSys '26}
}