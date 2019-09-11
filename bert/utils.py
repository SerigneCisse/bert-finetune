import time
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf


def load_ag_news_dataset(max_seq_len=512, test=False):
    """Loads the AG News corpus, which consists of news articles
    from the 4 largest classes in the AG’s corpus of news articles.
    The dataset contains 30,000 training examples for each class
    and 1,900 examples for each class for testing.
    https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
    """
    filename = "ag_news.zip"
    dataset_path = tf.keras.utils.get_file(filename,
                                           "https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/ag_news.zip",
                                           cache_subdir='datasets', extract=True)

    dataset_path = dataset_path.replace(".zip", "")

    try:
        if test:
            df = pd.read_csv(dataset_path+"/test.csv",
                             names=["label", "title", "article"], header=None)
        else:
            df = pd.read_csv(dataset_path+"/train.csv",
                             names=["label", "title", "article"], header=None)
            
    except Exception as e:
        print("Encounter weird pandas race condition", e)
        time.sleep(1)
        print("Attempting to read data again")
        if test:
            df = pd.read_csv(dataset_path+"/test.csv",
                             names=["label", "title", "article"], header=None)
        else:
            df = pd.read_csv(dataset_path+"/train.csv",
                             names=["label", "title", "article"], header=None)

    if test:
        titles = df["title"].tolist()
        raw_examples = df["article"].tolist()
    else:
        titles = df["title"].tolist()
        raw_examples = df["article"].tolist()
    
    examples = []
    len_list = []
    for i, raw in enumerate(raw_examples):
        raw = " ".join([str(titles[i]), str(raw)])
        raw = raw.replace("  ", " ").split(" ")
        len_list.append(len(raw))
        if len(raw) > max_seq_len:
            raw_cmb = raw[:max_seq_len//2] + raw[:-max_seq_len//2]
            raw = raw_cmb
        output = " ".join(raw)
        examples.append(output)
    examples = [" ".join(t.split(" ")[:max_seq_len]) for t in examples]
    examples = np.array(examples, dtype=object)[:, np.newaxis]

    if test:
        labels = df["label"].tolist()
    else:
        labels = df["label"].tolist()
    
    # map labels (1 ~ x) to classes (0 ~ x-1)
    labels = np.asarray(labels) - 1

    examples, labels = shuffle(examples, labels)

    num_classes = len(set(labels))

    if test:
        print("Loaded test set from:", dataset_path)
    else:
        print("Loaded training set from:", dataset_path)
    print("Examples:", len(examples), "Classes:", num_classes)
    #print(stats.describe(np.asarray(len_list)))
    
    return examples, labels, num_classes


def shard_dataset(examples, labels, hvd):
    """Split the dataset evenly among workers specified by hvd
    Args:
        examples (list): List of examples
        labels (list): List of labels
        hvd: Horovod instance
    Returns:
        examples (list): List of examples
        labels (list): List of labels
    """
    size = len(labels)
    shard_size = size//hvd.size()
    remainder = size % hvd.size()

    start_index = hvd.rank() * shard_size
    end_index = (hvd.rank() + 1) * shard_size + remainder

    examples, labels = examples[start_index:end_index], labels[start_index:end_index]
    examples, labels = shuffle(examples, labels)

    return examples, labels

def trunc_dataset(input_list_1, input_list_2, input_list_3, input_list_4, num_items):
    return input_list_1[:num_items], input_list_2[:num_items], input_list_3[:num_items], input_list_4[:num_items]
    
try:
    import GPUtil

    def get_gpu_vram():
        vram_list = []
        for gpu in GPUtil.getGPUs():
            vram_list.append(int(gpu.memoryTotal))
        return min(vram_list)
except Exception as e:
    print("[ERROR]", e)
    print("GPU VRAM detection not available")


def print_title(text):
    text = str(text).strip()
    print("="*len(text))
    print(text.upper())
    print("="*len(text))


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
