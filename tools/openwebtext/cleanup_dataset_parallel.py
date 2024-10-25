import ftfy
import json
from langdetect import detect
import time
import os
import sys
import multiprocessing
from tokenizer import Tokenizer

MIN_DOCUMENT_LENGHT = 128
CHUNK_SIZE = 10000  # 每个进程处理的行数块大小

temp_file_path = "/pscratch/sd/k/klhhhhh/openwebtext_data/"

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

def process_chunk(chunk, process_id, tokenizer_cache):
    """处理一个块，写入到单独的文件"""
    tokenizer = Tokenizer(cache_dir=tokenizer_cache)

    output_filename = temp_file_path + f"temp_output_{process_id}.jsonl"  # 每个进程的输出文件
    num_fixed_text = 0
    num_non_english_docs = 0
    chars_non_english_docs = 0
    num_small_docs = 0
    chars_small_docs = 0
    num_written_docs = 0
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for line in chunk:
            try:
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
                f.write(myjson_str + '\n')  # 每个结果写入文件
                num_written_docs += 1
            except Exception as e:
                print('    skipping ', e)

    # return info
    return {
        "num_written_docs": num_written_docs,
        "num_fixed_text": num_fixed_text,
        "num_non_english_docs": num_non_english_docs,
        "chars_non_english_docs": chars_non_english_docs,
        "num_small_docs": num_small_docs,
        "chars_small_docs": chars_small_docs,
        "output_filename": output_filename
    }

def filter_corpus_parallel(filename, out_filename, num_workers=12, chunk_size=10000, print_interval=10000):

    print(f' > filtering {filename} using {num_workers} workers')

    num_docs = 0
    num_written_docs = 0
    num_small_docs = 0
    num_fixed_text = 0
    num_non_english_docs = 0
    chars_non_english_docs = 0
    chars_small_docs = 0
    start_time = time.time()

    pool = multiprocessing.Pool(processes=num_workers)

    temp_files = []

    def collect_result(result):
        """主进程用于收集处理结果"""
        nonlocal num_docs, num_written_docs, num_fixed_text, num_non_english_docs
        nonlocal chars_non_english_docs, num_small_docs, chars_small_docs, temp_files

        num_written_docs += result['num_written_docs']
        num_fixed_text += result['num_fixed_text']
        num_non_english_docs += result['num_non_english_docs']
        chars_non_english_docs += result['chars_non_english_docs']
        num_small_docs += result['num_small_docs']
        chars_small_docs += result['chars_small_docs']
        temp_files.append(result['output_filename'])  # 收集临时文件名
        num_docs += result['num_written_docs']

        if num_docs % print_interval == 0:
            print_progress('[PROGRESS]', start_time, num_docs, num_fixed_text, num_non_english_docs,
                           chars_non_english_docs, num_small_docs, chars_small_docs)

    with open(filename, 'r') as fin:
        chunk = []
        process_id = 0  # 进程ID，用于生成唯一的文件名
        for line in fin:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                pool.apply_async(process_chunk, args=(chunk, process_id, './cache'), callback=collect_result)
                chunk = []
                process_id += 1
        if chunk:
            pool.apply_async(process_chunk, args=(chunk, process_id, './cache'), callback=collect_result)

        pool.close()
        pool.join()

    # 汇总所有子进程的文件
    with open(out_filename, 'wb') as output_file:
        for temp_file in temp_files:
            with open(temp_file, 'rb') as f:
                output_file.write(f.read())
            os.remove(temp_file)  # 汇总后删除临时文件

    print_progress('[FINAL]', start_time, num_docs, num_fixed_text, num_non_english_docs,
                   chars_non_english_docs, num_small_docs, chars_small_docs)

if __name__ == '__main__':

    print('building gpt2 dataset ...')

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    print(f'will be reading {input_filename}')
    print(f'and will write the results to {output_filename}')

    filter_corpus_parallel(input_filename, output_filename, num_workers=12)
