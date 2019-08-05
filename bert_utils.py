import numpy as np
from tqdm import tqdm

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

from bert.tokenization import FullTokenizer


class BERT(tf.keras.layers.Layer):
    """Wraps a BERT TFHub module in a Keras layer.
    Args:
        fine_tune_layers (int): Number of transformer cells to train.
        bert_path (str): URL to a module from TFHub.
          Use one listed on `https://tfhub.dev/s?q=bert`.
        output_size (int): Size of hidden layer and output.
          Defaults to 768 (BERTBASE), set to 1024 for BERTLARGE.
        return_sequence (bool): Use pooled output (classification), or
          sequence output (for character-level tasks).
        debug: Print debug output.
    Returns:
        BERT (tf.keras.layers.Layer): The constructed BERT layer.
    """

    def __init__(self, fine_tune_layers, bert_path, output_size=768, return_sequence=True, debug=False, **kwargs):
        self.fine_tune_layers = fine_tune_layers
        self.trainable = True
        self.output_size = output_size

        # bert_path uses one of the modules from
        # TFHub: https://tfhub.dev/s?q=bert
        self.bert_path = bert_path

        self.return_sequence = return_sequence
        self.debug = debug

        super(BERT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path,
                               trainable=self.trainable,
                               name="{}_module".format(self.name))

        trainable_vars = self.bert.variables

        t_vs = [var for var in trainable_vars if not "/cls/" in var.name]

        trainable_vars = t_vs

        layer_name_list = []

        for i, var in enumerate(trainable_vars):
            if "layer" in var.name:
                layer_name = var.name.split("/")[3]
                layer_name_list.append(layer_name)

        layer_names = list(set(layer_name_list))
        layer_names.sort()

        if self.debug:
            print(layer_names)

        if self.fine_tune_layers == -1:
            for var in trainable_vars:
                if self.return_sequence:
                    # do not use pooling layer
                    if "/pooler/" in var.name:
                        pass
                    else:
                        self._trainable_weights.append(var)
                else:
                    # use all layers
                    self._trainable_weights.append(var)

        else:
            # Select how many layers to fine tune
            last_n_layers = len(layer_names) - self.fine_tune_layers

            for var in trainable_vars:
                if "/pooler/" in var.name:
                    if self.return_sequence:
                        # we don't want to use it for sequence tasks
                        pass
                    elif self.fine_tune_layers == 0:
                        # nor if we don't want to train the whole model
                        pass
                    else:
                        self._trainable_weights.append(var)
                if "layer" in var.name:
                    layer_name = var.name.split("/")[3]
                    layer_num = int(layer_name.split("_")[1])+1
                    if layer_num > last_n_layers:
                        # Add to trainable weights
                        self._trainable_weights.append(var)

            if self.debug:
                print("BERT module loaded with", len(layer_names),
                      "Transformer cells, training all cells >", last_n_layers)

        if self.debug:
            print("DEBUG: Printing trainable vars")
            print("Name" + " "*48, "|", "Shape" + " "*8, "|", "Params")

        # Add non-trainable weights
        for i, var in enumerate(self.bert.variables):
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)
            else:
                if self.debug:
                    var_shape = var.get_shape()
                    var_params = 1
                    for dim in var_shape:
                        var_params *= dim
                    var_name = var.name.replace("bert_module/bert/", "")
                    var_name = var_name + str((52-len(var_name))*" ")
                    var_shape = str(var_shape) + \
                        str((13-len(str(var_shape)))*" ")
                    print(var_name, "|", var_shape, "|", var_params)

        super(BERT, self).build(input_shape)

    def call(self, inputs):
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=tf.cast(input_ids, dtype="int32"),
                           input_mask=tf.cast(input_mask, dtype="int32"),
                           segment_ids=tf.cast(segment_ids, dtype="int32"))
        if self.return_sequence:
            # [batch_size, sequence_length, hidden_size]
            result = self.bert(inputs=bert_inputs,
                               signature="tokens",
                               as_dict=True)["sequence_output"]
        else:
            # [batch_size, hidden_size]
            result = self.bert(inputs=bert_inputs,
                               signature="tokens",
                               as_dict=True)["pooled_output"]
        return result

    def compute_output_shape(self, input_shape):
        print("Input:", input_shape)
        if self.return_sequence:
            return (input_shape[1], self.output_size)
        else:
            return (self.output_size,)


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path, sess):
    """
    Get the vocab file and casing info from the Hub module.
    BERT doesn’t look at words as tokens. Rather, it looks at WordPieces.
    `tokenization.py` is the tokenizer that would turns your words into
    wordPieces appropriate for BERT.
    """
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info",
                                    as_dict=True)
    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                          tokenization_info["do_lower_case"], ])

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=512, verbose=False):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []

    if verbose:
        for example in tqdm(examples, desc="Converting examples to features"):
            input_id, input_mask, segment_id, label = convert_single_example(tokenizer,
                                                                             example,
                                                                             max_seq_length)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)
    else:
        for example in examples:
            input_id, input_mask, segment_id, label = convert_single_example(tokenizer,
                                                                             example,
                                                                             max_seq_length)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)

    return (np.array(input_ids), np.array(input_masks),
            np.array(segment_ids), np.array(labels).reshape(-1, 1),)


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(InputExample(guid=None,
                                          text_a=" ".join(text),
                                          text_b=None,
                                          label=label))
    return InputExamples
