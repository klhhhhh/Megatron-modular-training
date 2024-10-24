# Downloading datasets from hugging face

```
    git clone https://huggingface.co/datasets/Skylion007/openwebtext
```

# Extracting 21 tar to a directory

You need to specify your downloading datasets path, and the destination path.

1. Extract the .xz files from the downloading .tar files.
```
    python extract_xz_from_tar.py
``` 
2. Extract txt from the .xz files.
```
    python extract_txt_from_xz.py
```

# Merge txt to a json file.

```
    python merge_txt_to_json.py --data_path /path/to/txt/data --output_file /output/file/path
```

# Clean data.

```
    python cleanup_dataset_parallel.py /path/to/merged_output.json /path/to/self-define/cleaned_up.json | tee /path/to/output_file
```

