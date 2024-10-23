# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import ftfy
import json
from langdetect import detect
import numpy as np
import time
import os
import sys
from multiprocessing import Pool, cpu_count, Manager
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


def process_line(line, tokenizer):
    """处理每一行文档的函数，将结果返回给主进程"""
    try:
        myjson = json.loads(line)
        # 修复文本
        text = ftfy.fix_text(myjson['text'])
        is_fixed = text != myjson['text']
        myjson['text'] = text

        # 语言检测
        if detect(text) != 'en':
            return None, len(text), 'non_english', is_fixed

        # 检查文档长度
        if len(text) < (8 * MIN_DOCUMENT_LENGHT):
            tokens = tokenizer.tokenize_document(text)
            if len(tokens) < MIN_DOCUMENT_LENGHT:
                return None, len(text), 'small', is_fixed

        # 返回处理过的文档
        myjson = json.dumps(myjson, ensure_ascii=False)
        return myjson, len(text), 'valid', is_fixed

    except Exception as e:
        return None, 0, 'error', False


def filter_corpus_parallel(filename, out_filename, print_interval=10000, num_workers=1):
    if num_workers is None:
        num_workers = cpu_count()

    print(f'Starting parallel processing with {num_workers} workers.')

    tokenizer = Tokenizer(cache_dir='./cache')

    manager = Manager()
    num_docs = manager.Value('i', 0)
    num_written_docs = manager.Value('i', 0)
    num_small_docs = manager.Value('i', 0)
    num_fixed_text = manager.Value('i', 0)
    num_non_english_docs = manager.Value('i', 0)
    chars_non_english_docs = manager.Value('i', 0)
    chars_small_docs = manager.Value('i', 0)

    start_time = time.time()

    def update_progress(result):
        doc, length, status, is_fixed = result
        with num_docs.get_lock():
            num_docs.value += 1

        if is_fixed:
            with num_fixed_text.get_lock():
                num_fixed_text.value += 1

        if status == 'valid':
            with num_written_docs.get_lock():
                num_written_docs.value += 1
            return doc
        elif status == 'small':
            with num_small_docs.get_lock():
                num_small_docs.value += 1
            with chars_small_docs.get_lock():
                chars_small_docs.value += length
        elif status == 'non_english':
            with num_non_english_docs.get_lock():
                num_non_english_docs.value += 1
            with chars_non_english_docs.get_lock():
                chars_non_english_docs.value += length

        if num_docs.value % print_interval == 0:
            print_progress('[PROGRESS]', start_time, num_docs.value,
                           num_fixed_text.value, num_non_english_docs.value,
                           chars_non_english_docs.value, num_small_docs.value, chars_small_docs.value)
        return None

    with open(out_filename, 'wb') as f_out, open(filename, 'r') as f_in:
        pool = Pool(processes=num_workers)

        results = pool.imap(lambda line: process_line(line, tokenizer), f_in)
        for result in results:
            processed_doc = update_progress(result)
            if processed_doc:
                f_out.write(processed_doc.encode('utf-8'))
                f_out.write('\n'.encode('utf-8'))

        pool.close()
        pool.join()

    print_progress('[FINAL]', start_time, num_docs.value,
                   num_fixed_text.value, num_non_english_docs.value,
                   chars_non_english_docs.value, num_small_docs.value, chars_small_docs.value)


if __name__ == '__main__':

    print('Building GPT-2 dataset...')

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    print(f'Will be reading {input_filename}')
    print(f'And will write the results to {output_filename}')

    filter_corpus_parallel(input_filename, output_filename)
