import lzma
import os
import tarfile
from pathlib import Path
 
 
def decompress_xz_files(src_dir, dest_dir, start_index=1, end_index=1000):
    """Decompress .xz files containing multiple documents and copy each document to the destination directory."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for j in range(0,21):
        for i in range(start_index, end_index + 1):
            src_file = f"{src_dir}/urlsf_subset{j:02d}-{i}_data.xz"
            if os.path.exists(src_file):
                # Handle regular .xz files
                dest_file_path = os.path.join(dest_dir, f"extracted_content{j}-{i}.txt")
                with lzma.open(src_file, 'rt') as file:
                    content = file.read()
                with open(dest_file_path, 'w') as out_file:
                    out_file.write(content)
            else:
                print(f"File {src_file} does not exist")

        print(f"Decompressed and copied content of part {j:02d}")
 
# Specify your source and destination directories
source_directory = '/pscratch/sd/k/klhhhhh/openwebtext_data/openwebtext'
destination_directory = '/pscratch/sd/k/klhhhhh/openwebtext_data/txt_data'
 
# Call the function
decompress_xz_files(source_directory, destination_directory)
