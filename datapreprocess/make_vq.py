import glob
import os
import shutil
import io
import subprocess
import tempfile
import fsspec
import tarfile
import threading
from webdataset import ShardWriter
import random
import time
import zstandard as zstd
from contextlib import contextmanager
import argparse
from pathlib import Path
from transformers import GPTNeoXTokenizerFast
from braceexpand import braceexpand

import numpy as np


QUEUE_MAX = 10000
BUFFER_MIN = 100000
BUFFER_MAX = 200000
SHARD_SIZE = 8192
SLEEP_TIME = 1

def write_to_shard(chunks, shard_writer):
    for idx, chunk in enumerate(chunks):
        shard_writer.write({"__key__": f"{idx:012d}", "txt": str(chunk)})

def upload_to_s3_and_remove(fname, s3_path, chunk_size):
    fname_split = fname.split('/')
    s3_path = os.path.join(s3_path, f"{chunk_size - 1}", fname_split[-2], fname_split[-1])
    cmd = f'aws s3 cp {fname} {s3_path}'
    print('COMMAND:', cmd)
    if os.system(cmd) == 0:  # Check if the command was successful
        os.remove(fname)
        with open(f"{fname}.s3done", 'w') as marker_file:
            pass  # Create an empty marker file to indicate S3 upload is done


def process_files(file_list, buffer, buffer_lock, chunk_size, vocab_size):
    remaining_tokens = []
    queue = []

    def dump_queue_to_buffer():
        with buffer_lock:
            while queue:
                buffer.append(queue.pop(0))

    folder = "/".join(file_list[0].split("/")[0:-1])
    fs, output_path = fsspec.core.url_to_fs(folder)

    for file_name in file_list:
        print('Processing', file_name)

        try:
            with tempfile.TemporaryDirectory() as tempdir:
                tar_bytes = io.BytesIO(fs.open(file_name).read())
                with tarfile.open(fileobj=tar_bytes) as tar:
                    tar.extractall(tempdir)

                nps = glob.glob(os.path.join(tempdir, "*.npy")) 
                total_tokens = []
                for np_arr in nps:
                    vid = np.load(np_arr)
                    # Append EOF token
                    vid = np.hstack([vid, np.full((vid.shape[0], 1), vocab_size)])
                    vid = vid.reshape(-1)
                    total_tokens.append(vid)

                for tokens in total_tokens:
                    tokens = tokens.tolist()
                    for i in range(0, len(tokens), chunk_size):
                        chunk = tokens[i:i + chunk_size]
                        if len(chunk) < chunk_size:
                            remaining_tokens = chunk
                        else:
                            if len(buffer) > BUFFER_MAX:
                                time.sleep(1)
                                continue

                            if buffer_lock.locked():
                                if len(queue) < QUEUE_MAX:
                                    queue.append(chunk)
                                else:
                                    time.sleep(1)
                            else:
                                if queue:
                                    dump_queue_to_buffer()
                                with buffer_lock:
                                    buffer.append(chunk)
        except FileNotFoundError as e:
            print(f"{file_name} does not exist")



def consumer(my_id, output_dir, threads, buffer, buffer_lock, num_consumers, upload_to_s3=False, s3_path=None, chunk_size=2048):
    output_directory = f"{output_dir}/{chunk_size - 1}/{my_id}"
    os.makedirs(output_directory, exist_ok=True)
    shard_writer = ShardWriter(os.path.join(output_directory, "shard-%07d.tar"), maxcount=SHARD_SIZE)

    chunks = []

    start_time = time.time()

    while any(t.is_alive() for t in threads):
        time.sleep(SLEEP_TIME)
        with buffer_lock:
            lenb = len(buffer)
            print('Length of buffer', lenb)
            if lenb >= BUFFER_MIN:
                while buffer and len(chunks) < SHARD_SIZE:
                    random_index = random.randint(0, len(buffer) - 1)
                    chunks.append(buffer[random_index])
                    buffer.pop(random_index)  # Remove the selected element
        
        if len(chunks) == SHARD_SIZE:
            print(f'I am {my_id} and I am writing a shard.', len(buffer))
            write_to_shard(chunks, shard_writer)

            if upload_to_s3:
                upload_to_s3_and_remove(shard_writer.fname, s3_path, chunk_size)
            else:
                # Create a marker file to indicate that the shard is complete
                with open(f"{shard_writer.fname}.done", 'w') as marker_file:
                    pass  # Create an empty marker file

            chunks = []
            time_for_shard = time.time() - start_time
            print('shards / s', num_consumers / time_for_shard)
            print('tokens / s', num_consumers * SHARD_SIZE * chunk_size / time_for_shard)
            print('hours req for 1.2T tokens', 1_200_000_000_000 / (num_consumers * SHARD_SIZE * chunk_size / time_for_shard) / 3600)
        
            start_time = time.time()

    # Process the remaining items in the buffer after all threads have completed
    while buffer:
        with buffer_lock:
            while buffer and len(chunks) < SHARD_SIZE:
                random_index = random.randint(0, len(buffer) - 1)
                chunks.append(buffer[random_index])
                buffer.pop(random_index)  # Remove the selected element

        write_to_shard(chunks, shard_writer)


        if upload_to_s3:
            upload_to_s3_and_remove(shard_writer.fname, s3_path, chunk_size)
        else:
            # Create a marker file to indicate that the shard is complete
            with open(f"{shard_writer.fname}.done", 'w') as marker_file:
                pass  # Create an empty marker file
        chunks = []


def aligner_worker(output_dir, threads, num_consumers, upload_to_s3=False, s3_path=None, chunk_size=None):
    global_shard_id = 0
    tar_dir = f"{output_dir}/tars-{chunk_size - 1}"
    os.makedirs(tar_dir, exist_ok=True)
    marker_extension = ".s3done" if upload_to_s3 else ".done"

    # while any(t.is_alive() for t in threads):
    while True:
        time.sleep(SLEEP_TIME)
        no_markers = True

        for consumer_id in range(num_consumers):
            consumer_output_directory = f"{output_dir}/{chunk_size - 1}/{consumer_id}"

            if not os.path.exists(consumer_output_directory):
                continue

            marker_files = [f for f in os.listdir(consumer_output_directory) if f.endswith(marker_extension)]

            for marker_file in marker_files:
                no_markers = False
                shard_file = marker_file.replace(marker_extension, "")
                marker_path = os.path.join(consumer_output_directory, marker_file)

                # Only move the shard if the marker file exists
                if os.path.exists(marker_path):
                    if upload_to_s3:
                        s3_source_path = os.path.join(s3_path, f"{chunk_size - 1}/{consumer_id}/{shard_file}" )
                        destination_path = os.path.join(s3_path, f"tars-{chunk_size - 1}", f"shard-{global_shard_id:07d}.tar")
                        cmd = f'aws s3 cp {s3_source_path} {destination_path}'
                        os.system(cmd)
                    else:
                        source_path = os.path.join(consumer_output_directory, shard_file)
                        destination_path = os.path.join(tar_dir, f"shard-{global_shard_id:07d}.tar")
                        shutil.move(source_path, destination_path)

                    os.remove(marker_path)  # Remove the marker file
                    global_shard_id += 1

        if no_markers and not any(t.is_alive() for t in threads):
            break
    print("Aligner is done")


def main(input_files, output_dir, num_workers=32, num_consumers=8, upload_to_s3=False, s3_path=None, chunk_size=2048, vocab_size=1024):
    if "*" in input_files[0]:
        input_files = [glob.glob(input_file) for input_file in input_files]
        input_files = [x for y in input_files for x in y]
    elif "{" in input_files[0] and "}" in input_files[0]:
        input_files = [braceexpand(f) for f in input_files]
        input_files = [x for y in input_files for x in y]
    elif "s3://" in input_files[0]:
        cmd = f"aws s3 ls {input_files[0]} --recursive | grep '.tar$' | awk '{{print $4}}'"
        result = subprocess.check_output(cmd, shell=True).decode('utf-8')
        tars = [line.split("/")[-1] for line in result.splitlines()]
        input_files = [os.path.join(input_files[0], t) for t in tars]
    else:
        print("Some issue with input_files")
        return

    if upload_to_s3:
        assert s3_path is not None

    # Shuffle the input files
    random.shuffle(input_files)

    print('Input files', input_files)

    buffer = []  # Use list instead of queue.Queue
    buffer_lock = threading.Lock()

    files_per_worker = len(input_files) // num_workers
    threads = []
    for i in range(num_workers):
        start = i * files_per_worker
        end = (i + 1) * files_per_worker if i < num_workers - 1 else len(input_files)
        t = threading.Thread(target=process_files, args=(input_files[start:end], buffer, buffer_lock, chunk_size, vocab_size))
        t.start()
        threads.append(t)

    consumer_threads = []
    for i in range(num_consumers):
        t = threading.Thread(target=consumer, args=(i, output_dir, threads, buffer, buffer_lock, num_consumers, upload_to_s3, s3_path, chunk_size))
        t.start()
        consumer_threads.append(t)

    # Start the aligner worker thread
    aligner_thread = threading.Thread(target=aligner_worker, args=(output_dir, threads + consumer_threads, num_consumers, upload_to_s3, s3_path, chunk_size))
    aligner_thread.start()


    # for t in threads + consumer_threads + [aligner_thread]:
    for t in threads + consumer_threads:
        t.join()

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", type=str, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--num-consumers", type=int, default=8)
    parser.add_argument("--upload-to-s3", action='store_true')
    parser.add_argument("--s3-path", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--tokens-per-frame", type=int, default=256)
    parser.add_argument("--n-frames", type=int, default=16)

    args = parser.parse_args()

    chunk_size = args.n_frames * (args.tokens_per_frame + 1)

    main(args.input_files, args.output_dir, args.num_workers, args.num_consumers, args.upload_to_s3, args.s3_path, chunk_size, args.vocab_size)
