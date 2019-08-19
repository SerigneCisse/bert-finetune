import warnings
warnings.filterwarnings("ignore")

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

BERTLARGE     = False
USE_AMP       = True
USE_XLA       = True
MAX_SEQ_LEN   = 128
LEARNING_RATE = 2e-5
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

opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

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
                epochs=50, batch_size=56)

[eval_loss, eval_acc] = model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels, verbose=2, batch_size=112)

print("Loss:", eval_loss, "Acc:", eval_acc)

print("PHASE 2:")
print("Load and label whole dataset")

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

y_pred = model.predict([train_input_ids, train_input_masks, train_segment_ids], verbose=2, batch_size=112)
y_pred_class = np.argmax(y_pred, axis=1)
y_pred_class = y_pred_class.reshape(y_pred_class.shape[0], 1).shape

#pickle.dump(feat, open("./dataset.pickle", "wb"))

print("PHASE 3:")
print("Train model on labelled dataset")

log = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                y_pred_class, validation_data=test_set,
                workers=4, use_multiprocessing=True,
                verbose=2, callbacks=callbacks_list,
                epochs=50, batch_size=56)

[eval_loss, eval_acc] = model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels, verbose=2, batch_size=112)

print("End!")
print("Loss:", eval_loss, "Acc:", eval_acc)
