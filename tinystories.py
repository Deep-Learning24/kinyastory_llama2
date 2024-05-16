import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
import os
import random
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset

import pandas as pd
from tqdm import tqdm
import nltk

import re

from nltk.corpus import words

nltk.download('words')


DATA_CACHE_DIR = "data"
common_english_words = set(words.words())
# Define a regex pattern that keeps only English letters, numbers, basic punctuation, and spaces
pattern = re.compile(r"[^a-zA-Z0-9\s\.,!?;:()\'\"\-\—]")
def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://storytelling-m5a9.onrender.com/stories"
    data_filename = os.path.join(DATA_CACHE_DIR, "Kinyastories.json")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    print("Download done.")


def train_vocab(vocab_size, additional_directory=None):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset and
    additional data (if provided).

    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # Output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    

    # Read the JSON file containing the stories
    json_file = os.path.join(DATA_CACHE_DIR, "Kinyastories.json")
    with open(json_file, "r") as f:
        stories = json.load(f)

    # Read additional data files (if provided)
    additional_data = []
    if additional_directory:
        additional_files = os.listdir(additional_directory)
        for file_name in additional_files:
            if file_name == '.ipynb_checkpoints':
                continue
            file_path = os.path.join(additional_directory, file_name)
            if file_name.endswith('.csv'):
                # For CSV files, extract the content column
                df = pd.read_csv(file_path)
                if 'content' in df.columns:
                    additional_data.extend(df['content'].tolist())
            elif file_name == 'kin_community_2017_30K-sentences.txt':
                # For kin_community_2017_30K-sentences.txt, ignore the first part and extract text after the first tab
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) > 1:
                            additional_data.append(str(parts[1]))
                        else:
                            additional_data.append(str(parts[0]))
            else:
                # For other text files, read them like f.readlines()
                with open(file_path, 'r', encoding='utf-8') as f:
                    additional_data.extend(f.readlines())

    # Write each story and additional data to a temporary text file and train the tokenizer
    temp_text_file = os.path.join(DATA_CACHE_DIR, "temp_story.txt")
     

    with open(temp_text_file, "a", encoding="utf-8") as temp_f:
        for story in tqdm(stories):
            text = story["story_text"].strip()
            # Remove non-English characters
            text = pattern.sub('', text)
            # Filter out common English words and join the rest
            text = ' '.join([word for word in text.split() if word not in common_english_words])
            temp_f.write(text + '\n')

        for data in additional_data:
            # Convert the data to string, remove non-English characters, and filter words
            text = str(data).strip()
            text = pattern.sub('', text)
            text = ' '.join([word for word in text.split() if word not in common_english_words])
            temp_f.write(text + '\n')

    # Train the sentencepiece model using the temporary text file
    spm.SentencePieceTrainer.train(
        input=temp_text_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
    )

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")

    # Remove the temporary text file
    os.remove(temp_text_file)

def process_shard(json_file, vocab_size,additional_directory=None):
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    print("Tokenizer model: ", tokenizer_model)
    enc = Tokenizer(tokenizer_model)
    print("process_shard got called....")
    # Read the JSON file containing stories
    with open(json_file, "r") as f:
        stories = json.load(f)
        
    
    # Read additional data files (if provided)
    additional_data = []
    if additional_directory:
        additional_files = os.listdir(additional_directory)
        for file_name in additional_files:
            if file_name == '.ipynb_checkpoints':
                continue
            file_path = os.path.join(additional_directory, file_name)
            if file_name.endswith('.csv'):
                # For CSV files, extract the content column
                df = pd.read_csv(file_path)
                if 'content' in df.columns:
                    additional_data.extend(df['content'].tolist())
            elif file_name == 'kin_community_2017_30K-sentences.txt':
                # For kin_community_2017_30K-sentences.txt, ignore the first part and extract text after the first tab
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) > 1:
                            additional_data.append(parts[1])
                        else:
                            additional_data.append(parts[0])
            else:
                # For other text files, read them like f.readlines()
                with open(file_path, 'r', encoding='utf-8') as f:
                    additional_data.extend(f.readlines())
            
    all_tokens = []
    # Tokenize each story
    for i, story in enumerate(stories):
        text = story["story_text"]
        text = text.strip()  # get rid of leading/trailing whitespace
        text = pattern.sub('', text)
        text = ' '.join([word for word in text.split() if word not in common_english_words])
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        print(f"Encoded story {story['story_title']} ")

        # convert to uint16 nparray
        all_tokens.extend(tokens)
    # Tokenize the additional text (if provided)
    if additional_directory:
        for text in additional_data:
            text = pattern.sub('', str(text))
            text = ' '.join([word for word in text.split() if word not in common_english_words])
            tokens = enc.encode(str(text), bos=True, eos=False)
            all_tokens.extend(tokens)
        
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = os.path.join(DATA_CACHE_DIR, "all_stories.bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        tokenized_filename = os.path.join(bin_dir, "all_stories.bin")
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

def pretokenize(vocab_size):
    """
    Tokenizes each story in the TinyStories dataset.
    The tokenized files will be saved in the corresponding tok{N} directories,
    where N is the vocab size.
    """
    # Define the input JSON file containing stories
    json_file = os.path.join(DATA_CACHE_DIR, "Kinyastories.json")
    print("JSON file: ", json_file)

    # Create the output directory for tokenized files
    if vocab_size > 0:
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)
    print("Bin directory: ", bin_dir)
    
    process_shard(json_file, vocab_size,additional_directory=f"{DATA_CACHE_DIR}/kinyastories")

    print("Done.")



class PretokDataset(IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # Get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # Get DDP rank info
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # Combine the worker_id and worker_rank to create a unique seed for RNG
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with RNG seed {seed}")

        # Determine the bin file paths based on vocab source
        if self.vocab_source == "llama2":
            bin_dir = os.path.join(DATA_CACHE_DIR, "tok4096")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        
        print("Shared files: ", shard_filenames)

        # Train/validation split
        if self.split == "train":
            # Exclude a portion for validation
            total_data_size = os.path.getsize(shard_filenames[0])  # Get the size of the file in bytes
            print("total_data_size: ",total_data_size)
            validation_size = int(total_data_size * 0.1)  # 10% for validation
            with open(shard_filenames[0], "rb") as file:
                file.seek(validation_size)  # Move the file pointer to the validation split point
                train_data = file.read()
            print("Train data size: ", len(train_data))
        elif self.split == "val":
            # Take a portion for validation
            total_data_size = os.path.getsize(shard_filenames[0])  # Get the size of the file in bytes
           
            validation_size = int(total_data_size * 0.1)  # 10% for validation
            print("validation_size: ",validation_size)
            with open(shard_filenames[0], "rb") as file:
                val_data = file.read(validation_size)
            print("Validation data size: ", len(val_data))

        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}"

        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # Open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # Drop the last partial batch
                assert num_batches > 0, "This shard is way too small? Investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # Calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
   
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        # tokenizer_model = get_tokenizer_model_path(4096)
        # print("Tokenizer model: ", tokenizer_model)
        # enc = Tokenizer(tokenizer_model)
        # for i, (x, y) in enumerate(ds):
        #     if i == 5:  # Print only the first 5 samples
        #         break
        #     print("Sample Input:", enc.decode(x.tolist()))
        #     print("Sample Output:", enc.decode(y.tolist()))
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
    
        train_vocab(vocab_size=args.vocab_size,additional_directory=f"{DATA_CACHE_DIR}/kinyastories")
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
