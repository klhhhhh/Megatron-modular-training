import logging
import ftfy
import json
from langdetect import detect
import numpy as np
import time
import os
import sys
from multiprocessing import Pool, cpu_count, current_process
from tokenizer import Tokenizer

MIN_DOCUMENT_LENGHT = 128

logging.basicConfig(filename='/pscratch/sd/k/klhhhhh/openwebtext_data/output.log', level=logging.INFO, 
                    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

def process_line(line):
    try:
        # 
        process_name = current_process().name
        pid = os.getpid()

        # 
        myjson = json.loads(line)
        text = ftfy.fix_text(myjson['text'])
        if detect(text) != 'en':
            logging.info(f"Process {process_name} (PID: {pid}) Skipping non-English document")
            return None, None, None  # Skip non-English documents

        # 检查文档长度
        if len(text) < (8 * MIN_DOCUMENT_LENGHT):
            tokenizer = Tokenizer(cache_dir='./cache')
            tokens = tokenizer.tokenize_document(text)
            if len(tokens) < MIN_DOCUMENT_LENGHT:
                logging.info(f"Process {process_name} (PID: {pid}) Skipping small document")
                return None, None, None  # Skip small documents

        myjson['text'] = text
        logging.info(f"Process {process_name} (PID: {pid}) Processed document successfully")
        return json.dumps(myjson, ensure_ascii=False), len(text), 'valid'
    except Exception as e:
        logging.error(f"Process {process_name} (PID: {pid}) Error processing line: {e}")
        return None, None, 'error'

def filter_corpus_parallel(filename, out_filename, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPU cores

    logging.info(f'Starting parallel processing with {num_workers} workers.')

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
            pool = Pool(num_workers)
            results = pool.imap(process_line, fin)

            for result, length, status in results:
                num_docs += 1
                if status == 'valid':
                    f.write(result.encode('utf-8'))
                    f.write('\n'.encode('utf-8'))
                    num_written_docs += 1
                elif status == 'small':
                    num_small_docs += 1
                    chars_small_docs += length
                elif status == 'non-english':
                    num_non_english_docs += 1
                    chars_non_english_docs += length

                if num_docs % 10000 == 0:
                    logging.info(f'[PROGRESS] {num_docs} documents processed.')

    logging.info(f'[FINAL] {num_docs} documents processed.')


if __name__ == '__main__':
    print('building gpt2 dataset ...')

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    logging.info(f'will be reading {input_filename}')
    logging.info(f'and will write the results to {output_filename}')

    filter_corpus_parallel(input_filename, output_filename, 12)
