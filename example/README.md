# Usage
 Change Env in `example/paths.sh`, `example/model-name/config.sh`
| Env | Description |
| --- | ----------- |
|PYTHONPATH|Path to Megatron-LM
|STALLOC_DIR|Parent path of STAlloc
|DATA_PATH,VOCAB_FILE,MERGE_FILE,TOKENIZER_PATH|Data

## Run Experiments (E2-E5)
### E2: 
```shell
cd example/llama
bash llama_stalloc.sh
cd example/gpt2
bash gpt2_stalloc.sh
```

### E3:
```shell
cd example/llama
bash llama_torch.sh
bash llama_torch_es.sh
# in GMLake docker
bash llama_gmlake.sh

cd example/gpt2
bash gpt2_torch.sh
bash gpt2_torch_es.sh
# in GMLake docker
bash gpt2_gmlake.sh
```

### E4:
```shell
cd example/qwen-moe
bash qwen_stalloc.sh
```

### E5:
```shell
cd example/qwen-moe
bash qwen_torch.sh
bash qwen_torch_es.sh
# in GMLake docker
bash qwen_gmlake.sh
```