import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import pickle
import multiprocessing
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.compat.v1.keras import layers

from tqdm import tqdm

import utils
import bert_utils
import bert_optimizer

BERTLARGE     = False
USE_AMP       = True
USE_XLA       = True
MAX_SEQ_LEN   = 128
LEARNING_RATE = 1e-5
TUNE_LAYERS   = -1

DATASET_PORTION = float(os.environ["DATASET_PORTION"])

if BERTLARGE:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
    H_SIZE = 1024
else:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    H_SIZE = 768

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if USE_XLA:
    opt_level = tf.OptimizerOptions.ON_1
    tf.enable_resource_variables()
else:
    opt_level = tf.OptimizerOptions.OFF
config.graph_options.optimizer_options.global_jit_level = opt_level
config.graph_options.rewrite_options.auto_mixed_precision = USE_AMP
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# ### Create Tokenizer

tokenizer = bert_utils.create_tokenizer_from_hub_module(BERT_PATH, sess)

# ### Preprocess Data

train_text, train_label, num_classes = utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                                  test=False)

if DATASET_PORTION < 1:
    num_examples = int(len(train_label) * DATASET_PORTION)
    _, train_text, _, train_label= train_test_split(train_text, train_label, test_size=DATASET_PORTION, stratify=train_label)
else:
    num_examples = len(train_label)

train_label = np.asarray(train_label)
train_examples = bert_utils.convert_text_to_examples(train_text, train_label)

feat = bert_utils.convert_examples_to_features(tokenizer,
                                               train_examples,
                                               max_seq_length=MAX_SEQ_LEN,
                                               verbose=False)

(train_input_ids, train_input_masks, train_segment_ids, train_labels) = feat

print("Number of training examples:", len(train_labels))

examples, labels, num_classes = utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                           test=True)
labels = np.asarray(labels)
test_examples = bert_utils.convert_text_to_examples(examples, labels)
feat = bert_utils.convert_examples_to_features(tokenizer,
                                               test_examples,
                                               max_seq_length=MAX_SEQ_LEN,
                                               verbose=False)

(test_input_ids, test_input_masks, test_segment_ids, test_labels) = feat

test_input_ids, test_input_masks, test_segment_ids, test_labels = shuffle(test_input_ids,
                                                                          test_input_masks,
                                                                          test_segment_ids,
                                                                          test_labels)

test_set = ([test_input_ids, test_input_masks, test_segment_ids], test_labels)

# ## Build Keras Model

class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

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
l_drop = MCDropout(rate=0.5)(l_bert)
out_pred = layers.Dense(num_classes, activation="softmax")(l_drop)

model = tf.keras.models.Model(inputs=in_bert, outputs=out_pred)

opt = bert_optimizer.RAdam(lr=LEARNING_RATE)

if USE_AMP:
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

model.summary()

# ## Train Model

def scheduler(epoch):
    warmup_steps = 26000
    warmup_epochs = warmup_steps//num_examples
    if epoch < warmup_epochs:
        return LEARNING_RATE*(epoch/warmup_epochs)
    else:
        return LEARNING_RATE

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

callbacks_list = [lr_schedule, early_stop]

print("PHASE 1:")
print("Train on 'golden' portion of dataset")

log = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                train_labels, validation_data=test_set,
                workers=4, use_multiprocessing=True,
                verbose=2, callbacks=callbacks_list,
                epochs=1000, batch_size=56)

[eval_loss, eval_acc] = model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels, verbose=2, batch_size=256)

print("Loss:", eval_loss, "Acc:", eval_acc)

print("PHASE 2:")
print("Load whole dataset")

train_text, train_label, num_classes = utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                                  test=False)

train_label = np.asarray(train_label)
train_examples = bert_utils.convert_text_to_examples(train_text, train_label)

feat = bert_utils.convert_examples_to_features(tokenizer,
                                               train_examples,
                                               max_seq_length=MAX_SEQ_LEN,
                                               verbose=False)

(train_input_ids, train_input_masks, train_segment_ids, train_labels) = feat    

print("Number of training examples:", len(train_labels))

print("PHASE 3:")
print("Train model on labelled dataset")

LEARNING_RATE = 1e-6

def scheduler_2(epoch):
    warmup_epochs = 1
    if epoch < warmup_epochs:
        return LEARNING_RATE
    else:
        return LEARNING_RATE/epoch

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_2)
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

callbacks_list = [lr_schedule, early_stop]

for i in range(5):
    print("\nIteration " + str(i) + " :\n")

    print("Generating predictions with MCDropout")
    y_pred_list = []
    for _ in tqdm(range(10)):
        y_pred = model.predict([train_input_ids, train_input_masks, train_segment_ids], verbose=2, batch_size=256)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_pred_list.append(y_pred_class)
    agg_y_pred = np.stack(y_pred_list, axis=-1)
    
    # try to find wrong predictions
    acc = 0
    wrong = 0
    corr_guess = 0
    false_pos = 0
    false_neg = 0
    for i, truth in tqdm(enumerate(train_label), total=len(train_label)):
        pred = agg_y_pred[i, :]
        m = stats.mode(pred).mode
        correct = truth==m
        diff = 0
        for item in pred:
            if item != m:
                diff += 1
        if diff > max(1,(5-i)):
            same = False
        else:
            same = True
        if correct and same:
            # correct guess as correct
            acc += 1
        if correct and not same:
            # wrong guess as wrong
            false_pos += 1
            wrong += 1
        if not correct and same:
            # wrong guess as correct
            false_neg += 1
            wrong += 1
        if not correct and not same:
            # correct guess as wrong
            corr_guess += 1
            acc += 1
    print("Correct guesses:", acc)
    print("Wrong guesses:", wrong, ";", round(wrong/len(train_label)*100,1), "%")
    print("Correct guess as wrong:", corr_guess)
    print("Wrong guess as wrong:", false_pos)
    print("Total labels for human:", corr_guess+false_pos)
    
    # generate fixed labels
    y_pred_fixed = []
    for i in range(len(train_label)):
        pred = agg_y_pred[i, :]
        m = stats.mode(pred).mode
        diff = 0
        for item in pred:
            if item != m:
                diff += 1
        if diff > 5:
            same = False
        else:
            same = True
        if same:
            y_pred_fixed.append(m)
        else:
            y_pred_fixed.append(train_label[i])
    y_pred_fixed = np.asarray(y_pred_fixed)

    log = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                    y_pred_fixed, validation_data=test_set,
                    workers=4, use_multiprocessing=True,
                    verbose=2, callbacks=callbacks_list,
                    epochs=50, batch_size=56)

    [eval_loss, eval_acc] = model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels, verbose=2, batch_size=256)

    print("Loss:", eval_loss, "Acc:", eval_acc)

print("End!")
