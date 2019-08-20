# ======================
# Command Line Arguments
# ======================

import time

non_train_start = time.time()

import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""Train a news classifier for the AG News dataset.

Intended to be run with Horovod for distributed training:
horovodrun -np $N_GPU -H localhost:$N_GPU python3 news_classification.py""")

parser.add_argument("--xla",
                    help="Use XLA to speed up model training",
                    action="store_true")
parser.add_argument("--amp",
                    help="Use Mixed Precision to speed up model training",
                    action="store_true")
parser.add_argument("--bertlarge",
                    help="Use BERTLARGE instead of BERTBASE",
                    action="store_true")
parser.add_argument("--lazyadam",
                    help="Use LazyAdam optimizer instead of Adam",
                    action="store_true")
parser.add_argument("--radam",
                    help="Use RAdam optimizer instead of Adam",
                    action="store_true")
parser.add_argument("--fp16_allreduce",
                    help="Use FP16 compression for allreduce",
                    action="store_true")
parser.add_argument("--sparse_as_dense",
                    help="Use convert sparse tensors to dense tensors for allreduce",
                    action="store_true")
parser.add_argument("--save_weights",
                    help="Save weight checkpoints",
                    action="store_true")
parser.add_argument("--progress",
                    help="Show live model training progress",
                    action="store_true")
parser.add_argument("--dev",
                    help="Only load a truncated dataset",
                    action="store_true")
parser.add_argument("--debug",
                    help="Print TensorFlow debug output",
                    action="store_true")
parser.add_argument("--profiling",
                    help="Only run one example. Used for profiling kernels.",
                    action="store_true")
parser.add_argument("--epochs",
                    help="Number of epochs to train for",
                    type=int)
parser.add_argument("--batch_size",
                    help="Batch size to use for training",
                    type=int)
parser.add_argument("--lr",
                    help="Learning Rate to use for training",
                    type=float)
parser.add_argument("--finetune",
                    help="Number of transformer layers to train",
                    type=int)
parser.add_argument("--maxseqlen",
                    help="Maximum input sequence length",
                    type=int)
parser.add_argument("--results",
                    help="Output directory for results")
parser.add_argument("--weights",
                    help="Path to load weights from")

args = parser.parse_args()

if args.epochs:
    N_EPOCHS = args.epochs
else:
    N_EPOCHS = 2
    
if args.finetune is not None:
    TUNE_LAYERS = args.finetune
else:
    TUNE_LAYERS = -1

if args.results:
    RESULTS_DIR = args.results
else:
    RESULTS_DIR = "/results/"
    
if args.weights:
    WEIGHTS_PATH = args.weights
else:
    WEIGHTS_PATH = False

if args.maxseqlen:
    MAX_SEQ_LEN = args.maxseqlen
else:
    MAX_SEQ_LEN = 512
    
import os
from pathlib import Path
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.compat.v1.keras import layers
import horovod.tensorflow as hvd
import horovod.tensorflow.keras as hvd_keras

import utils
import bert_utils
import bert_optimizer

# ==================
# Initialize Horovod
# ==================

hvd.init()
time.sleep(0.5)

# ==========================
# Training Script Parameters
# ==========================

# ensure trailing slash
if RESULTS_DIR[-1] != "/":
    RESULTS_DIR = RESULTS_DIR+"/"

if hvd.rank() == 0:
    utils.ensure_dir(RESULTS_DIR)
    utils.ensure_dir("./dbpedia/")
else:
    time.sleep(0.5)

if args.bertlarge:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
    H_SIZE = 1024
else:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    H_SIZE = 768

if args.batch_size:
    # use provided batch size
    BATCH_SIZE = args.batch_size
else:
    # use pre-determined batch size for current task
    if utils.get_gpu_vram() > 17000:
        if args.bertlarge:
            BATCH_SIZE = 12
        else:
            BATCH_SIZE = 56
    else:
        if args.bertlarge:
            BATCH_SIZE = 2
        else:
            BATCH_SIZE = 21

# ====================================================
# Create TensorFlow Session before loading BERT module
# ====================================================

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "2"

config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.gpu_options.allow_growth = True
config.gpu_options.Experimental.use_numa_affinity = True
if args.xla:
    opt_level = tf.OptimizerOptions.ON_1
    tf.enable_resource_variables()
else:
    opt_level = tf.OptimizerOptions.OFF
config.graph_options.optimizer_options.global_jit_level = opt_level
config.graph_options.rewrite_options.auto_mixed_precision = args.amp
config.intra_op_parallelism_threads = multiprocessing.cpu_count()//hvd.size()
config.inter_op_parallelism_threads = multiprocessing.cpu_count()//hvd.size()
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

tokenizer = bert_utils.create_tokenizer_from_hub_module(BERT_PATH, sess)

# ============
# Load Dataset
# ============

if hvd.rank() == 0:
    print_progress = True
else:
    print_progress = False

time.sleep(hvd.rank())

train_text, train_label, num_classes = utils.load_dbpedia_dataset(max_seq_len=MAX_SEQ_LEN,
                                                                  suffix=str(hvd.rank()),
                                                                  test=False)

# load, preprocess and save data to pickle

# training set

train_feat_cache = "./dbpedia/train_feat.pickle."+str(hvd.rank())
train_feat = Path(train_feat_cache)

if train_feat.is_file():
    feat = pickle.load(open(train_feat_cache, "rb"))
else:
    train_text, train_label = utils.shard_dataset(train_text, train_label, hvd)
    train_label = np.asarray(train_label)
    train_examples = bert_utils.convert_text_to_examples(train_text, train_label)
    feat = bert_utils.convert_examples_to_features(tokenizer,
                                                   train_examples,
                                                   max_seq_length=MAX_SEQ_LEN,
                                                   verbose=print_progress)
    pickle.dump(feat, open(train_feat_cache, "wb"))

(train_input_ids, train_input_masks, train_segment_ids, train_labels) = feat

train_input_ids, train_input_masks, train_segment_ids, train_labels = shuffle(train_input_ids,
                                                                              train_input_masks,
                                                                              train_segment_ids,
                                                                              train_labels)

if args.dev:
    # for quicker testing during development, we reduce the dataset size
    num_items = len(train_labels)//100
    train_input_ids, train_input_masks, train_segment_ids, train_labels = utils.trunc_dataset(train_input_ids,
                                                                                              train_input_masks,
                                                                                              train_segment_ids,
                                                                                              train_labels)
    print("(Truncated dataset) Number of training examples:", num_items)
    
if args.profiling:
    num_items = 1
    train_input_ids, train_input_masks, train_segment_ids, train_labels = utils.trunc_dataset(train_input_ids,
                                                                                              train_input_masks,
                                                                                              train_segment_ids,
                                                                                              train_labels)
    print("(Truncated dataset) Number of training examples:", num_items)
    
# test set

test_feat_cache = "./dbpedia/test_feat.pickle."+str(hvd.rank())
test_feat = Path(test_feat_cache)

if test_feat.is_file():
    feat = pickle.load(open(test_feat_cache, "rb"))
else:
    examples, labels, num_classes = utils.load_dbpedia_dataset(max_seq_len=MAX_SEQ_LEN,
                                                               suffix=str(hvd.rank()),
                                                               test=True)
    examples, labels = utils.shard_dataset(examples, labels, hvd)
    labels = np.asarray(labels)
    test_examples = bert_utils.convert_text_to_examples(examples, labels)
    feat = bert_utils.convert_examples_to_features(tokenizer,
                                                   test_examples,
                                                   max_seq_length=MAX_SEQ_LEN,
                                                   verbose=print_progress)
    pickle.dump(feat, open(test_feat_cache, "wb"))

(test_input_ids, test_input_masks, test_segment_ids, test_labels) = feat

test_input_ids, test_input_masks, test_segment_ids, test_labels = shuffle(test_input_ids,
                                                                          test_input_masks,
                                                                          test_segment_ids,
                                                                          test_labels)

test_set = ([test_input_ids, test_input_masks, test_segment_ids], test_labels)

if args.dev:
    num_items = len(test_labels)//10
    test_input_ids, test_input_masks, test_segment_ids, test_labels = utils.trunc_dataset(test_input_ids,
                                                                                          test_input_masks,
                                                                                          test_segment_ids,
                                                                                          test_labels)
    print("(Truncated dataset) Number of test examples:", num_items)

if args.profiling:
    num_items = 1
    test_input_ids, test_input_masks, test_segment_ids, test_labels = utils.trunc_dataset(test_input_ids,
                                                                                          test_input_masks,
                                                                                          test_segment_ids,
                                                                                          test_labels)
    print("(Truncated dataset) Number of test examples:", num_items)
    
# =================
# Build Keras model
# =================

if args.amp:
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

# =======================
# Create the training job
# =======================

if args.fp16_allreduce:
    cmp = hvd_keras.Compression.fp16
else:
    cmp = hvd_keras.Compression.none

if args.lr:
    LEARNING_RATE = args.lr
else:
    LEARNING_RATE = 3e-5
    
opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

if args.lazyadam:
    opt = bert_optimizer.LazyAdam(lr=LEARNING_RATE)
    
if args.radam:
    opt = bert_optimizer.RAdam(lr=LEARNING_RATE)

opt = hvd_keras.DistributedOptimizer(opt,
                                     compression=cmp,
                                     sparse_as_dense=args.sparse_as_dense)

if args.amp:
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

if WEIGHTS_PATH:
    model.load_weights(WEIGHTS_PATH)
    
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

if hvd.rank() == 0:
    model.summary()

# ==============
# Start training
# ==============

rank_prefix = "[" + str(hvd.rank()) + "]"
utils.print_title(rank_prefix + " Started Training")

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# Broadcast initial variable states from rank 0 to all other processes:

bcast = hvd.broadcast_global_variables(0)
bcast.run(session=sess)

if hvd.rank() == 0:
    if args.progress:
        verbose = 1
    else:
        verbose = 2
else:
    verbose = 0

if not args.profiling:
    callbacks = [
        # average metrics among workers after every epoch
        hvd_keras.callbacks.MetricAverageCallback(),
        hvd_keras.callbacks.LearningRateWarmupCallback(warmup_epochs=1, verbose=verbose),
        hvd_keras.callbacks.LearningRateScheduleCallback(start_epoch=1, end_epoch=10,
                                                         staircase=True,
                                                         multiplier=bert_optimizer.inv_decay),
    ]
else:
    callbacks = []

if hvd.rank() == 0:
    if args.save_weights:
        checkpoint_path = RESULTS_DIR+"./weights.best.h5"
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                              monitor='val_acc', verbose=1,
                                                              save_best_only=True,
                                                              save_weights_only=True)
        callbacks.append(model_checkpoint)
        
non_train_end = time.time()

train_start_time = time.time()

log = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                train_labels, validation_data=test_set,
                workers=4, use_multiprocessing=True,
                verbose=verbose, callbacks=callbacks,
                epochs=N_EPOCHS, batch_size=BATCH_SIZE)

train_end_time = time.time()

if hvd.rank() == 0:
    log_name = str(hvd.rank())+"_log.csv"

    res_acc = log.history["acc"]
    res_loss = log.history["loss"]
    res_val_acc = log.history["val_acc"]
    res_val_loss = log.history["val_loss"]

    dict_log = {"acc": res_acc,
                "loss": res_loss,
                "val_acc": res_val_acc,
                "val_loss": res_val_loss}

    df = pd.DataFrame(dict_log)
    df.to_csv(RESULTS_DIR+log_name)
    
    print("| Non-Train Time | Train Time |")
    print("| ", int(non_train_end-non_train_start), " | ", int(train_end_time-train_start_time), " |")

    f = open(RESULTS_DIR+str(train_end_time-train_start_time)+".time", "w")
    f.write(str(train_end_time-train_start_time))
    f.close()

rank_prefix = "[" + str(hvd.rank()) + "]"
utils.print_title(rank_prefix + " Finished Successfully!")
