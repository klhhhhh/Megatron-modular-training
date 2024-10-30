import os

output_dir = '/pscratch/sd/k/klhhhhh/openwebtext_data'
out_filename = 'cleaned_up.json'
file_num = 32

with open(os.path.join(output_dir, out_filename), 'wb') as output_file:
    for i in range(file_num):
        temp_file_name = f"temp_output_{i}.jsonl"
        temp_file = os.path.join(output_dir, temp_file_name)
        with open(temp_file, 'rb') as f:
            output_file.write(f.read())

        print("write"+temp_file_name)

