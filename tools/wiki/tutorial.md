# 1. 下载数据集

1. Download the Wikipedia compressed dataset ([enwiki-latest-pages-articles.xml.bz2](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2))

2. Use the [wikiextractor](https://github.com/attardi/wikiextractor) tool to decompress the dataset

```shell
pip install wikiextractor
python -m wikiextractor.WikiExtractor --json enwiki-latest-pages-articles.xml.bz2
```

After decompression, you will get a folder `text` with the following structure:

```bash
text
├── AA
    ├── wiki_00
    ├── wiki__01
    ├── ...
├── AB
├── AC
├── AD
├── AE
├── ...
├── GD
└── GE
```

The folder contains multiple subfolders, each containing multiple JSON formatted datasets. Each `wiki_00` is actually a JSON formatted file.

3. Preprocess the decompressed dataset

When training GPT, the decompressed dataset cannot be used directly. We need to preprocess the dataset in the `text` directory using the [tools/preprocess_data.py](Megatron-DeepSpeed/tools/preprocess_data.py) provided by Megatron-Deepspeed. This will generate two binary files with `bin` and `idx` extensions.

However, `tools/preprocess_data.py` can only process a single JSON file, while in the second step we have hundreds of thousands of JSON files. What should we do? One way to handle this is to merge all the JSON files from the third step into a single JSON file, and then preprocess the merged file. Before processing, you need to run the following commands to download the GPT-related files, which are mainly used for preprocessing.

```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

After downloading, execute the following code:

```bash
#!/bin/bash  
    
# Set the ROOT path  
ROOT="/data/personal/nus-hx/Wikipedia/text"  

# Check if the wiki_all.json file exists, if so, delete it  
if [ -f "$ROOT/wiki_all.json" ]; then  
        rm "$ROOT/wiki_all.json"  
fi  
    
# Create an empty wiki_all.json file  
touch "$ROOT/wiki_all.json"
    
# Traverse all files in the ROOT path  
find $ROOT -type f -name "*" -print0 | while IFS= read -r -d $'\0' file; do  
        # Append all file contents to the wiki_all.json file  
        cat "$file" >> "$ROOT/wiki_all.json"  
done  

cd /path/to/Megatron-Deepspeed
python tools/preprocess_data.py \
--input "$ROOT/wiki_all.json" \
--output-prefix my-gpt2 \
--dataset-impl mmap \
--tokenizer-type GPT2BPETokenizer   \
--append-eod  \
--vocab-file gpt2-vocab.json \
--merge-file gpt2-merges.txt  \
--workers 16 \
--partitions 16
```

reference: https://github.com/NVIDIA/Megatron-LM/issues/117

# 2. Run the code

```
bash ./examples/pretrain_gpt.sh
```

