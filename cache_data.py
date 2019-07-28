import tensorflow.compat.v1 as tf

"""
# Huffpost Dataset

TRAIN_SET_URL = "https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/huffnews/news_train.csv.zip"
TEST_SET_URL = "https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/huffnews/news_test.csv.zip"

train_dataset = tf.keras.utils.get_file("news_train.csv.zip", TRAIN_SET_URL,
                                       cache_subdir='datasets', extract=True)

test_dataset = tf.keras.utils.get_file("news_test.csv.zip", TEST_SET_URL,
                                       cache_subdir='datasets', extract=True)
"""

# AG News Dataset
filename = "ag_news.zip"
dataset_path = tf.keras.utils.get_file(filename,
                                       "https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/ag_news.zip",
                                       cache_subdir='datasets', extract=True)
