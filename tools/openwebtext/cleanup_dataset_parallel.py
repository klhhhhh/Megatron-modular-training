# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import ftfy
import json
from langdetect import detect
import numpy as np
import time
import os
import sys
import concurrent.futures

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
    try:
        myjson = json.loads(line)
        # Fix text
        text = ftfy.fix_text(myjson['text'])
        if text != myjson['text']:
            fixed_text = True
        else:
            fixed_text = False
        myjson['text'] = text
        # Detect language.
        if detect(text) != 'en':
            print('[non-english text]', myjson)
            return None, None, len(text), False, False
        # On average each token is 5 characters so 8 is an upper bound.
        if len(text) < (8 * MIN_DOCUMENT_LENGHT):
            tokens = tokenizer.tokenize_document(text)
            if len(tokens) < MIN_DOCUMENT_LENGHT:
                print('[small document, skipping]:', myjson)
                return None, None, len(text), False, True
        myjson = json.dumps(myjson, ensure_ascii=False)
        return myjson.encode('utf-8'), fixed_text, len(text), False, False
    except Exception as e:
        print(f'    skipping ', line, e)
        return None, None, 0, False, False

def filter_corpus(filename, out_filename, print_interval=100, max_workers=1):

    print(f' > filtering {filename} with {max_workers} threads')

    tokenizer = Tokenizer(cache_dir='./cache')

    num_docs = 0
    num_written_docs = 0
    num_small_docs = 0
    num_fixed_text = 0
    num_non_english_docs = 0
    chars_non_english_docs = 0
    chars_small_docs = 0
    start_time = time.time()

    with open(out_filename, 'wb') as f:
        with open(filename, 'r') as fin:
            # Use ThreadPoolExecutor to parallelize processing lines
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_line, line, tokenizer): line for line in fin}
                for future in concurrent.futures.as_completed(futures):
                    num_docs += 1
                    result = future.result()
                    if result:
                        myjson, fixed_text, text_length, is_non_english, is_small = result

                        if myjson:
                            f.write(myjson)
                            f.write('\n'.encode('utf-8'))
                            num_written_docs += 1
                        if fixed_text:
                            num_fixed_text += 1
                        if is_non_english:
                            num_non_english_docs += 1
                            chars_non_english_docs += text_length
                        if is_small:
                            num_small_docs += 1
                            chars_small_docs += text_length
                    if num_docs % print_interval == 0:
                        print_progress('[PROGRESS]', start_time, num_docs,
                                       num_fixed_text, num_non_english_docs,
                                       chars_non_english_docs, num_small_docs,
                                       chars_small_docs)

    print_progress('[FINAL]', start_time, num_docs,
                   num_fixed_text, num_non_english_docs,
                   chars_non_english_docs, num_small_docs, chars_small_docs)

if __name__ == '__main__':

    print('building gpt2 dataset ...')

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    max_workers = 12

    print(f'will be reading {input_filename}')
    print(f'and will write the results to {output_filename}')
    print(f'using {max_workers} threads')

    filter_corpus(input_filename, output_filename, max_workers=max_workers)
