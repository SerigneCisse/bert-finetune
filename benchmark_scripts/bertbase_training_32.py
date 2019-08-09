#!/usr/bin/env python
# coding: utf-8

# # BERT Training
# 
# Single GPU BERT Training Notebook


# In[2]:

import os
from pathlib import Path
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.compat.v1.keras import layers

import utils
import bert_utils
import bert_optimizer


# ## Notebook Parameters

# In[3]:


BERTLARGE     = False
USE_AMP       = True
USE_XLA       = True
MAX_SEQ_LEN   = 512
LEARNING_RATE = 1e-5
TUNE_LAYERS   = -1


# In[4]:


if BERTLARGE:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
    H_SIZE = 1024
else:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    H_SIZE = 768


# ## Create TensorFlow session

# In[5]:


os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "2"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.Experimental.use_numa_affinity = True
if USE_XLA:
    opt_level = tf.OptimizerOptions.ON_1
    tf.enable_resource_variables()
else:
    opt_level = tf.OptimizerOptions.OFF
config.graph_options.optimizer_options.global_jit_level = opt_level
config.graph_options.rewrite_options.auto_mixed_precision = USE_AMP
config.intra_op_parallelism_threads = multiprocessing.cpu_count()//2
config.inter_op_parallelism_threads = multiprocessing.cpu_count()//2
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


# ## Load Dataset

# ### Create Tokenizer

# In[6]:


tokenizer = bert_utils.create_tokenizer_from_hub_module(BERT_PATH, sess)


# ### Preprocess Data

# In[7]:


train_text, train_label, num_classes = utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                                  test=False)

train_label = np.asarray(train_label)
train_examples = bert_utils.convert_text_to_examples(train_text, train_label)
feat = bert_utils.convert_examples_to_features(tokenizer,
                                               train_examples,
                                               max_seq_length=MAX_SEQ_LEN,
                                               verbose=True)

(train_input_ids, train_input_masks, train_segment_ids, train_labels) = feat

train_input_ids, train_input_masks, train_segment_ids, train_labels = shuffle(train_input_ids,
                                                                              train_input_masks,
                                                                              train_segment_ids,
                                                                              train_labels)


# In[8]:


examples, labels, num_classes = utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                           test=True)
labels = np.asarray(labels)
test_examples = bert_utils.convert_text_to_examples(examples, labels)
feat = bert_utils.convert_examples_to_features(tokenizer,
                                               test_examples,
                                               max_seq_length=MAX_SEQ_LEN,
                                               verbose=True)

(test_input_ids, test_input_masks, test_segment_ids, test_labels) = feat

test_input_ids, test_input_masks, test_segment_ids, test_labels = shuffle(test_input_ids,
                                                                          test_input_masks,
                                                                          test_segment_ids,
                                                                          test_labels)

test_set = ([test_input_ids, test_input_masks, test_segment_ids], test_labels)


# ## Build Keras Model

# In[9]:


if USE_AMP:
    tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')

in_id = layers.Input(shape=(MAX_SEQ_LEN,), name="input_ids")
in_mask = layers.Input(shape=(MAX_SEQ_LEN,), name="input_masks")
in_segment = layers.Input(shape=(MAX_SEQ_LEN,), name="segment_ids")

in_bert = [in_id, in_mask, in_segment]

l_bert = bert_utils.BERT(fine_tune_layers=TUNE_LAYERS,
                         bert_path=BERT_PATH,
                         return_sequence=False,
                         output_size=H_SIZE,
                         debug=False)(in_bert)

out_pred = layers.Dense(num_classes, activation="softmax")(l_bert)

model = tf.keras.models.Model(inputs=in_bert, outputs=out_pred)


# In[10]:


opt = bert_optimizer.LazyAdam(lr=LEARNING_RATE)

if USE_AMP:
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")


# In[11]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])


# In[12]:


model.summary()


# ## Train Model

# In[13]:


log = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                train_labels, validation_data=test_set,
                workers=4, use_multiprocessing=True,
                verbose=2, callbacks=[],
                epochs=3, batch_size=42)

