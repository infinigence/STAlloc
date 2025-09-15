# Usage
Preprcess dataset with the tool in Megatron-LM:
```shell
python Megatron-LM/tools/preprocess_data.py \
       --input /path/to/data.jsonl \
       --output-prefix my-prefix \
       --vocab-file /path/to/gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file /path/to/gpt2-merges.txt \
       --workers 8 \
       --append-eod
```
This will generate `my-prefix_text_document.idx` and `my-prefix_text_document.bin` in your current directory. Please add `/path/to/my-prefix_text_document` to `DATA_PATH` in `example/paths.sh`.

 Change Env in `example/paths.sh`, `example/model-name/config.sh`
| Env | Description |
| --- | ----------- |
|PYTHONPATH|Path to Megatron-LM
|STALLOC_DIR|Parent path of STAlloc
|DATA_PATH,VOCAB_FILE,MERGE_FILE,TOKENIZER_PATH|Data

## Run Experiments (E2-E5)
### E2: 
```shell
# in NGC PyTorch 24.03 docker
cd example/llama
bash llama_stalloc.sh
cd example/gpt2
bash gpt2_stalloc.sh
```

### E3:
```shell
# in NGC PyTorch 24.03 docker
cd example/llama
bash llama_torch.sh
bash llama_torch_es.sh
# in GMLake docker
bash llama_gmlake.sh

# in NGC PyTorch 24.03 docker
cd example/gpt2
bash gpt2_torch.sh
bash gpt2_torch_es.sh
# in GMLake docker
bash gpt2_gmlake.sh
```

Get statistics:
```shell
python example/memory_statistics.py example/analyze/llama/ example/analyze/gpt2/
```

### E4:
```shell
# in NGC PyTorch 24.03 docker
cd example/qwen-moe
bash qwen_stalloc.sh
```

### E5:
```shell
# in NGC PyTorch 24.03 docker
cd example/qwen-moe
bash qwen_torch.sh
bash qwen_torch_es.sh
# in GMLake docker
bash qwen_gmlake.sh
```

Get statistics:
```shell
python example/memory_statistics.py example/analyze/qwen-moe/
```

## Note
Offloading is not supported in the open-source test version of Megatron-LM. Therefore, we implemented and tested this feature (ZOR in Figure 7 of paper) in our private Megatron repository. For this set of experiments, we provide the original experimental data.