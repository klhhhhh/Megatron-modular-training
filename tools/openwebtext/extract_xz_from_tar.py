import lzma
import os
import tarfile
from pathlib import Path
 
 
def decompress_tar_files(src_dir, dest_dir, start_index=0, end_index=20):
    """Decompress .tar files containing multiple documents and copy each document to the destination directory."""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for i in range(start_index, end_index + 1):
        src_file = f"{src_dir}/urlsf_subset{i:02d}.tar"
        if os.path.exists(src_file):
            # Check if the file is a tarball
            if tarfile.is_tarfile(src_file):
                # Open the tarball file
                with tarfile.open(src_file, mode='r') as tar:
                    tar.extractall(path=dest_dir)
                    print(f"Extracted all contents of {src_file} to {dest_dir}")
        else:
            print(f"File {src_file} does not exist")
 

# Specify your source and destination directories
source_directory = '/pscratch/sd/k/klhhhhh/openwebtext/subsets'
destination_directory = '/pscratch/sd/k/klhhhhh/openwebtext_data'
 
# Call the function
decompress_tar_files(source_directory, destination_directory)
