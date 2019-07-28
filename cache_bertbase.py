# BERTBASE

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import layers
import bert_utils

BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
MAX_SEQ_LEN = 512

in_id = layers.Input(shape=(MAX_SEQ_LEN,), name="input_ids")
in_mask = layers.Input(shape=(MAX_SEQ_LEN,), name="input_masks")
in_segment = layers.Input(shape=(MAX_SEQ_LEN,), name="segment_ids")

in_bert = [in_id, in_mask, in_segment]

l_bert = bert_utils.BERT(fine_tune_layers=-1,
                         bert_path=BERT_PATH,
                         return_sequence=False,
                         output_size=768,
                         debug=False)(in_bert)
