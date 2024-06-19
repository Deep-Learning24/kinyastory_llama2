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
pattern = re.compile("[^a-zA-Z0-9\s\.,!?;:()\'\"\-\—]")
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
             # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove special characters like #, @
            text = re.sub(r'[@#]', '', text)
            temp_f.write(text + '\n')

        for data in additional_data:
            # Convert the data to string, remove non-English characters, and filter words
            text = str(data).strip()
            text = pattern.sub('', text)
            text = ' '.join([word for word in text.split() if word not in common_english_words])
             # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove special characters like #, @
            text = re.sub(r'[@#]', '', text)
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
         # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters like #, @
        text = re.sub(r'[@#]', '', text)
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
             # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove special characters like #, @
            text = re.sub(r'[@#]', '', text)
            
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
    
    @staticmethod
    def get_num_samples(split, max_seq_len, vocab_size, vocab_source):
        if vocab_source == "llama2":
            bin_dir = os.path.join(DATA_CACHE_DIR, "tok4096")
        elif vocab_source == "custom":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        
        elif vocab_source == "flan_dataset":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"flan{vocab_size}")
        
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        
            

        total_samples = 0
        for shard in shard_filenames:
            shard_size = os.path.getsize(shard)
            num_samples = shard_size // (max_seq_len * 2)  # 2 bytes per uint16 token
            total_samples += num_samples

        if split == "val":
            total_samples = int(total_samples * 0.1)  # 10% for validation

        return total_samples
    
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
        elif self.vocab_source == "flan_dataset":
            bin_dir = os.path.join(DATA_CACHE_DIR, f"flan{self.vocab_size}")
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

                    

class FinetuningDataPreprocessor:
    def __init__(self):
        pass
    def generate_flan_examples(self, story_text: str, prompt_preambles: list) -> list:
        return [f"{prompt_preamble} : {story_text}" for prompt_preamble in prompt_preambles]
    
    def clean_text(self, text: str) -> str:
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters like #, @
        text = re.sub(r'[@#]', '', text)
        # Remove other unwanted patterns
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        # Remove common English words
        text = ' '.join([word for word in text.split() if word.lower() not in common_english_words])
        return text.strip()
    
    def prepare_flan_dataset(self, json_file: str) -> list:
        print("Preparing the flan dataset examples")
        
        with open(json_file, "r") as f:
            stories = json.load(f)
        
        prompt_preambles = [
            "ncira umugani aho", "ncir'umugani aho", "ncir'umugan'aho", "ncir'umugani aho",
            "ncira umugani wa", "ncir'umugani wa", "ncir'umugani wa", "ncir'umugani wa",
            "ncir'umugani aho", "ncira umugani w", "ncir'umugani w", "ncir'umugani w",
            "ncir'umugani w", "mbwira inkuru ya", "mbwira'inkuru ya", "mbwira inkuru y", "mbwira'inkuru y"
        ]
        
        generated_dataset = []
        for i, story in enumerate(stories):
            story_context = f"{story['story_title']} ,{story['themes']}"
            story_text = f"{story_context} : inkuru: {story['story_text']}"
            story_text = self.clean_text(story_text)
            
            generated_dataset.extend(self.generate_flan_examples(story_text, prompt_preambles))
        
        return generated_dataset
    
    def save_generated_dataset(self, generated_dataset: list, output_file: str):
        with open(output_file, "w") as f:
            json.dump(generated_dataset, f, ensure_ascii=False, indent=4)
        print(f"Generated dataset saved to {output_file}")
    
    
    def tokenize_flan_dataset(self,json_file,vocab_size):
        
        tokenizer_model = get_tokenizer_model_path(vocab_size)
        enc = Tokenizer(tokenizer_model)
        
        all_tokens=[]
        generated_dataset = self.prepare_flan_dataset(json_file)
        self.save_generated_dataset(generated_dataset,  os.path.join(DATA_CACHE_DIR, "flan_stories.json"))
        
        for story in generated_dataset:
            tokens = enc.encode(str(story), bos=True, eos=False)
            all_tokens.extend(tokens)
            
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        
            # calculate the output filename
        if vocab_size == 0:
            # if we're using Llama 2, just save the tokenized file in the same dir
            tokenized_filename = os.path.join(DATA_CACHE_DIR, "flan_stories.bin")
        else:
            # save .bin files into a new tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"flan{vocab_size}")
            os.makedirs(bin_dir, exist_ok=True)
            tokenized_filename = os.path.join(bin_dir, "flan_stories.bin")
        # write the bytes
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
        print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

        
        
def pretokenize_flan_dataset(vocab_size):
    """
    Tokenizes each story in the TinyStories dataset.
    The tokenized files will be saved in the corresponding tok{N} directories,
    where N is the vocab size.
    """
    # Define the input JSON file containing stories
    json_file = os.path.join(DATA_CACHE_DIR, "Kinyastories.json")
    finetuningDataPreprocessor = FinetuningDataPreprocessor()
    finetuningDataPreprocessor.tokenize_flan_dataset(json_file,vocab_size)
    

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
    @staticmethod
    def get_train_dataset_size(**dataset_kwargs):
        return PretokDataset.get_num_samples(**dataset_kwargs)
    
        

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
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab","pretokenize_flan"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
    
        train_vocab(vocab_size=args.vocab_size,additional_directory=f"{DATA_CACHE_DIR}/kinyastories")
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize_flan":
        pretokenize_flan_dataset(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
