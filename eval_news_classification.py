# ======================
# Command Line Arguments
# ======================

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--xla",
                    help="Use XLA to speed up model training",
                    action="store_true")
parser.add_argument("--amp",
                    help="Use Mixed Precision to speed up model training",
                    action="store_true")
parser.add_argument("--bertlarge",
                    help="Use Mixed Precision to speed up model training",
                    action="store_true")
parser.add_argument("--weights",
                    help="Path to load weights from")

args = parser.parse_args()

if args.xla:
    USE_XLA = True
else:
    USE_XLA = False

if args.amp:
    MIXED_PRECISION = True
else:
    MIXED_PRECISION = False

if args.bertlarge:
    BERTLARGE = True
else:
    BERTLARGE = False

if args.weights:
    WEIGHTS_PATH = args.weights
else:
    WEIGHTS_PATH = "./weights.best.h5"

import time

import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.keras import layers

import utils
import bert_utils

# ======================
# Eval Script Parameters
# ======================

if BERTLARGE:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
    H_SIZE = 1024
else:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    H_SIZE = 768

MAX_SEQ_LEN = 128

TUNE_LAYERS = -1

# ==========================
# Create TensorFlow Session
# before loading BERT module
# ==========================

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if USE_XLA:
    opt_level = tf.OptimizerOptions.ON_1
    tf.enable_resource_variables()
else:
    opt_level = tf.OptimizerOptions.OFF
config.graph_options.optimizer_options.global_jit_level = opt_level
config.graph_options.rewrite_options.auto_mixed_precision = MIXED_PRECISION
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

examples, labels, num_classes = utils.load_ag_news_dataset(max_seq_len=MAX_SEQ_LEN,
                                                           test=True)

tokenizer = bert_utils.create_tokenizer_from_hub_module(BERT_PATH, sess)

test_examples = bert_utils.convert_text_to_examples(examples, labels)

feat = bert_utils.convert_examples_to_features(tokenizer,
                                               test_examples,
                                               max_seq_length=MAX_SEQ_LEN,
                                               verbose=True)

(test_input_ids, test_input_masks, test_segment_ids, test_labels) = feat

# =====================
# Build the Keras model
# =====================

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

opt = tf.keras.optimizers.Adam()

if MIXED_PRECISION:
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
tf.keras.backend.set_session(sess)

model.load_weights(WEIGHTS_PATH)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

model.summary()

start_time = time.time()

results = model.evaluate([test_input_ids, test_input_masks, test_segment_ids],
                         test_labels, verbose=1, batch_size=64)

end_time = time.time()

print(results)
