import ftfy
import json
from langdetect import detect
import time
import os
import sys
import multiprocessing
from tokenizer import Tokenizer

MIN_DOCUMENT_LENGHT = 128

def print_progress(prefix, start_time, num_docs, num_fixed_text,
                   num_non_english_docs, chars_non_english_docs,
                   num_small_docs, chars_small_docs):

    string = prefix + ' | '
    string += 'elapsed time: {:.2f} | '.format(time.time() - start_time)
    string += 'documents: {} | '.format(num_docs)
    string += 'fixed text: {} | '.format(num_fixed_text)
    string += 'non-english: {} | '.format(num_non_english_docs)
    string += 'non-english chars: {} | '.format(chars_non_english_docs)
    string += 'small docs: {} | '.format(num_small_docs)
    string += 'small docs chars: {}'.format(chars_small_docs)
    print(string, flush=True)

def process_lines(start_line, end_line, filename, process_id, tokenizer_cache, output_dir, result_queue):
    
    print(f"Process {process_id} started")

    tokenizer = Tokenizer(cache_dir= output_dir + tokenizer_cache)

    output_filename = os.path.join(output_dir, f"temp_output_{process_id}.jsonl")
    num_fixed_text = 0
    num_non_english_docs = 0
    chars_non_english_docs = 0
    num_small_docs = 0
    chars_small_docs = 0
    num_written_docs = 0

    with open(output_filename, 'w', encoding='utf-8') as f_out:
        with open(filename, 'r', encoding='utf-8') as f_in:
            counter = 0
            for i, line in enumerate(f_in):
                if i < start_line:
                    continue  # 跳过未到达的行
                if i >= end_line:
                    break  # 已超出需要处理的行
                try:
                    counter += 1
                    if counter%3000 == 0:
                        print(f"process {process_id} dealt with {counter} lines.")
                    myjson = json.loads(line)
                    # Fix text
                    text = ftfy.fix_text(myjson['text'])
                    if text != myjson['text']:
                        num_fixed_text += 1
                    myjson['text'] = text
                    # Detect language.
                    if detect(text) != 'en':
                        num_non_english_docs += 1
                        chars_non_english_docs += len(text)
                        continue
                    # On average each token is 5 characters, so 8 is an upper bound.
                    if len(text) < (8 * MIN_DOCUMENT_LENGHT):
                        tokens = tokenizer.tokenize_document(text)
                        if len(tokens) < MIN_DOCUMENT_LENGHT:
                            num_small_docs += 1
                            chars_small_docs += len(text)
                            continue
                    myjson_str = json.dumps(myjson, ensure_ascii=False)
                    f_out.write(myjson_str + '\n')  # 每个结果写入文件
                    num_written_docs += 1
                except Exception as e:
                    # print('    skipping ', e)
                    pass

    print(f"Process {process_id} finished: {num_written_docs} documents written.")

    result_queue.put({
        "num_written_docs": num_written_docs,
        "num_fixed_text": num_fixed_text,
        "num_non_english_docs": num_non_english_docs,
        "chars_non_english_docs": chars_non_english_docs,
        "num_small_docs": num_small_docs,
        "chars_small_docs": chars_small_docs,
        "output_filename": output_filename
    })

def get_file_line_count(filename):
    """计算文件总行数"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def filter_corpus_parallel(filename, out_filename, num_workers=32, print_interval=10000):

    print(f' > filtering {filename} using {num_workers} workers')

    total_lines = get_file_line_count(filename)
    lines_per_worker = total_lines // num_workers
    remaining_lines = total_lines % num_workers

    num_docs = 0
    num_written_docs = 0
    num_small_docs = 0
    num_fixed_text = 0
    num_non_english_docs = 0
    chars_non_english_docs = 0
    chars_small_docs = 0
    start_time = time.time()

    # 临时文件存储目录
    output_dir = '/pscratch/sd/k/klhhhhh/openwebtext_data'
    os.makedirs(output_dir, exist_ok=True)

    processes = []
    temp_files = []
    result_queue = multiprocessing.Queue()

    start_line = 0
    for process_id in range(num_workers):
        # 每个进程的行范围
        end_line = start_line + lines_per_worker + (1 if process_id < remaining_lines else 0)

        # 创建并启动进程
        p = multiprocessing.Process(target=process_lines, args=(start_line, end_line, filename, process_id, './cache', output_dir, result_queue))
        processes.append(p)
        p.start()

        # 更新进程处理的行范围
        temp_files.append(os.path.join(output_dir, f"temp_output_{process_id}.jsonl"))
        start_line = end_line

    # 等待所有进程结束
    for p in processes:
        p.join()

    while not result_queue.empty():
        result = result_queue.get()
        num_written_docs += result['num_written_docs']
        num_fixed_text += result['num_fixed_text']
        num_non_english_docs += result['num_non_english_docs']
        chars_non_english_docs += result['chars_non_english_docs']
        num_small_docs += result['num_small_docs']
        chars_small_docs += result['chars_small_docs']

    print_progress('[FINAL]', start_time, num_written_docs, num_fixed_text, num_non_english_docs,
                   chars_non_english_docs, num_small_docs, chars_small_docs)


if __name__ == '__main__':

    print('building gpt2 dataset ...')

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    print(f'will be reading {input_filename}')
    print(f'and will write the results to {output_filename}')

    filter_corpus_parallel(input_filename, output_filename, num_workers=32)
