# Command Line Arguments

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--bertbase",
                    help="Cache BERTBASE TF Hub module",
                    action="store_true")

parser.add_argument("--bertlarge",
                    help="Cache BERTLARGE TF Hub module",
                    action="store_true")

parser.add_argument("--agnews",
                    help="Cache AG News Dataset",
                    action="store_true")

args = parser.parse_args()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.compat.v1.keras import layers
import bert_utils

"""
# Huffpost Dataset

TRAIN_SET_URL = "https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/huffnews/news_train.csv.zip"
TEST_SET_URL = "https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/huffnews/news_test.csv.zip"

train_dataset = tf.keras.utils.get_file("news_train.csv.zip", TRAIN_SET_URL,
                                       cache_subdir='datasets', extract=True)

test_dataset = tf.keras.utils.get_file("news_test.csv.zip", TEST_SET_URL,
                                       cache_subdir='datasets', extract=True)
"""

MAX_SEQ_LEN = 512

if args.bertbase:
    print("[INFO ] Caching BERTBASE")
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    in_id = layers.Input(shape=(MAX_SEQ_LEN,), name="input_ids")
    in_mask = layers.Input(shape=(MAX_SEQ_LEN,), name="input_masks")
    in_segment = layers.Input(shape=(MAX_SEQ_LEN,), name="segment_ids")
    in_bert = [in_id, in_mask, in_segment]
    l_bert = bert_utils.BERT(fine_tune_layers=-1,
                             bert_path=BERT_PATH,
                             return_sequence=False,
                             output_size=768,
                             debug=False)(in_bert)
    
if args.bertlarge:
    print("[INFO ] Caching BERTLARGE")
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
    in_id = layers.Input(shape=(MAX_SEQ_LEN,), name="input_ids")
    in_mask = layers.Input(shape=(MAX_SEQ_LEN,), name="input_masks")
    in_segment = layers.Input(shape=(MAX_SEQ_LEN,), name="segment_ids")
    in_bert = [in_id, in_mask, in_segment]
    l_bert = bert_utils.BERT(fine_tune_layers=-1,
                             bert_path=BERT_PATH,
                             return_sequence=False,
                             output_size=1024,
                             debug=False)(in_bert)
    
if args.agnews:
    print("[INFO ] Caching AG News Dataset")
    filename = "ag_news.zip"
    dataset_path = tf.keras.utils.get_file(filename,
                                           "https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/ag_news.zip",
                                           cache_subdir='datasets', extract=True)
