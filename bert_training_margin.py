import os
import copy
from pathlib import Path
import pickle
import multiprocessing
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.WARN)
from tensorflow.compat.v1.keras import layers

from tqdm import tqdm

import bert.model
import bert.utils
import bert.optimizer

BERTLARGE      = False
USE_AMP        = True
USE_XLA        = True
MAX_SEQ_LEN    = 128
LEARNING_RATE  = 1e-5
TUNE_LAYERS    = -1
DROPOUT_RATE   = 0.5
BATCH_SIZE     = 48
EVAL_BATCHSIZE = 256
VAL_FREQ       = 3

DATASET_PORTION = float(os.environ["DATASET_PORTION"])

if BERTLARGE:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
    H_SIZE = 1024
else:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    H_SIZE = 768

sess = tf.Session()
tf.keras.backend.set_session(sess)

# Create Tokenizer

tokenizer = bert.model.create_tokenizer_from_hub_module(BERT_PATH, sess)

# Preprocess Data

train_text, train_label, num_classes = bert.utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                                       test=False)

if DATASET_PORTION < 1:
    num_examples = int(len(train_label) * DATASET_PORTION)
    _, train_text, _, train_label= train_test_split(train_text, train_label, test_size=DATASET_PORTION, stratify=train_label)
else:
    num_examples = len(train_label)
    
golden_dataset = (train_text, train_label)

#train_label = np.asarray(train_label)
feat = bert.model.convert_text_to_features(tokenizer, train_text, train_label, max_seq_length=MAX_SEQ_LEN, verbose=False)
(train_input_ids, train_input_masks, train_segment_ids, train_labels) = feat

print("Number of training examples:", len(train_labels))

examples, labels, num_classes = bert.utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                                test=True)
#labels = np.asarray(labels)
feat = bert.model.convert_text_to_features(tokenizer, examples, labels, max_seq_length=MAX_SEQ_LEN, verbose=False)
(test_input_ids, test_input_masks, test_segment_ids, test_labels) = feat

test_set = ([test_input_ids, test_input_masks, test_segment_ids], test_labels)

# Build Keras Model

def create_model():
    tf.keras.backend.clear_session()
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
    
    if USE_AMP:
        tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')

    in_id = layers.Input(shape=(MAX_SEQ_LEN,), name="input_ids")
    in_mask = layers.Input(shape=(MAX_SEQ_LEN,), name="input_masks")
    in_segment = layers.Input(shape=(MAX_SEQ_LEN,), name="segment_ids")
    in_bert = [in_id, in_mask, in_segment]
    l_bert = bert.model.BERT(fine_tune_layers=TUNE_LAYERS,
                             bert_path=BERT_PATH,
                             return_sequence=True,
                             output_size=H_SIZE,
                             debug=False)(in_bert)
    l_bert = layers.Reshape((MAX_SEQ_LEN, H_SIZE))(l_bert)
    l_drop_1 = layers.SpatialDropout1D(rate=DROPOUT_RATE)(l_bert)
    l_conv = layers.Conv1D(H_SIZE//2, 1)(l_drop_1)
    l_flat = layers.Flatten()(l_conv)
    l_drop_2 = layers.Dropout(rate=DROPOUT_RATE)(l_flat)
    out_pred = layers.Dense(num_classes, activation="softmax")(l_drop_2)

    model = tf.keras.models.Model(inputs=in_bert, outputs=out_pred)

    opt = bert.optimizer.RAdam(lr=LEARNING_RATE)

    if USE_AMP:
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    model.summary()
    
    return model
    
model = create_model()

# Train Model

def scheduler(epoch):
    warmup_steps = 30000
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
                validation_freq=VAL_FREQ, verbose=2, callbacks=callbacks_list,
                epochs=1000, batch_size=BATCH_SIZE)

[eval_loss, eval_acc] = model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels, verbose=2, batch_size=EVAL_BATCHSIZE)

print("Loss:", eval_loss, "Acc:", eval_acc)

print("PHASE 2:")
print("Load whole dataset")

train_text, train_label, num_classes = bert.utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                                       test=False)

#train_label = np.asarray(train_label)
feat = bert.model.convert_text_to_features(tokenizer, train_text, train_label, max_seq_length=MAX_SEQ_LEN, verbose=False)
(train_input_ids, train_input_masks, train_segment_ids, train_labels) = feat

print("Number of training examples:", len(train_labels))

print("PHASE 3:")
print("Train model on labelled dataset")

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

relabel_iterations = 5
num_pred_iterations = 10
threshold = 0.3

for i in range(relabel_iterations):
    print("\nIteration ", i+1, "of", relabel_iterations, "iterations:\n")

    print("Generating predictions with Dropout")
    y_softmax_list = []
    for _ in range(num_pred_iterations):
        with tf.keras.backend.learning_phase_scope(1):
            y_pred = model.predict([train_input_ids, train_input_masks, train_segment_ids], verbose=2, batch_size=EVAL_BATCHSIZE)
        y_softmax_list.append(y_pred)
    
    agg_y_softmax = np.stack(y_softmax_list, axis=-2)
    probs = np.sum(agg_y_softmax, axis=1)/num_pred_iterations
    probs = probs.tolist()

    margin_list = []

    false_positive = 0
    positive = 0
    
    try:
        train_label_ = golden_dataset[1].tolist()
    except Exception as e:
        print("Ignoring error:", e)
    try:
        train_text_ = golden_dataset[0].tolist()
    except Exception as e:
        print("Ignoring error:", e)

    for i, prob in enumerate(probs):
        # margin: top_pred - (top-1)_pred
        copy_prob = copy.deepcopy(prob)
        copy_prob.sort()
        margin = copy_prob[-1] - copy_prob[0]
        if margin < threshold:
            # for our own scoring purpose
            margin_list.append(margin)
            pred = np.argmax(prob)
            if pred == train_label[i]:
                false_positive += 1
            else:
                positive += 1
            # add corrected example into golden dataset
            if train_text[i] not in train_text_:
                train_text_.append(train_text[i])
                train_label_.append(train_label[i])
                
    golden_dataset = (train_text_, train_label_)
            
    acc = []
    for i, prob in enumerate(probs):
        pred = np.argmax(prob)
        if pred == train_label[i]:
            acc.append(1)
        else:
            acc.append(0)
    acc = sum(acc)/len(probs)

    print("Threshold <", threshold)
    print("Correct wrong:", positive)
    print("Incorrect wrong:", false_positive)
    print("Useful effort:", positive/(positive+false_positive))
    print("Overall accuracy:", acc)
    
    # regenerate dataset
    
    train_text_, train_label_ = golden_dataset[0], golden_dataset[1]
    
    num_examples = len(train_label_)

    #train_label_ = np.asarray(train_label_)
    feat = bert.model.convert_text_to_features(tokenizer, train_text_, train_label_, max_seq_length=MAX_SEQ_LEN, verbose=False)
    (train_input_ids_, train_input_masks_, train_segment_ids_, train_labels_) = feat
    
    model = create_model()
    
    def scheduler_2(epoch):
        warmup_steps = 30000
        warmup_epochs = warmup_steps//num_examples
        if epoch < warmup_epochs:
            return LEARNING_RATE*(epoch/warmup_epochs)
        else:
            return LEARNING_RATE

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler_2)
    callbacks_list = [lr_schedule, early_stop]
    
    # retrain model
    log = model.fit([train_input_ids_, train_input_masks_, train_segment_ids_],
                train_labels_, validation_data=test_set,
                validation_freq=VAL_FREQ, verbose=2, callbacks=callbacks_list,
                epochs=1000, batch_size=BATCH_SIZE)
    
    [eval_loss, eval_acc] = model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels, verbose=2, batch_size=EVAL_BATCHSIZE)

    print("Loss:", eval_loss, "Acc:", eval_acc)
    
    total_relabel = positive + false_positive
    if total_relabel < 200:
        threshold += 0.1

print("End!")
