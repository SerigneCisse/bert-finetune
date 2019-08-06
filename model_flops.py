import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

# set up TensorFlow session

config = tf.ConfigProto()
# allow us to instrument the GPU VRAM usage
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# key parameters

MAX_SEQ_LEN = 512
BERTLARGE = False

if BERTLARGE:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"
    H_SIZE = 1024
else:
    BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    H_SIZE = 768
    
# create synthetic data

bert_inputs = dict(input_ids=tf.ones(shape=(1,MAX_SEQ_LEN), dtype=tf.int32, name="input_tokens"),
                   input_mask=tf.ones(shape=(1,MAX_SEQ_LEN), dtype=tf.int32, name="input_mask"),
                   segment_ids=tf.ones(shape=(1,MAX_SEQ_LEN), dtype=tf.int32, name="input_padding"))

fake_labels = tf.ones(shape=(1,1), dtype=tf.int32, name="fake_labels")

bert_module = hub.Module(BERT_PATH,
                         trainable=True,
                         name="bert_module")

fwd_prop = bert_module(bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
fwd_prop = tf.nn.xw_plus_b(fwd_prop,
                           tf.ones(shape=(H_SIZE,1), dtype=tf.float32, name="fake_weights"),
                           tf.ones(shape=(1), dtype=tf.float32, name="fake_bias"),
                           name="matmul_final")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fwd_prop, labels=fake_labels))
optimizer = tf.train.AdamOptimizer().minimize(cost)

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_meta = tf.RunMetadata()

# run forward+backward pass

_, _ = sess.run([optimizer, cost], options=run_options, run_metadata=run_meta)

# get FLOPS

opts = tf.profiler.ProfileOptionBuilder.float_operation()    
flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd="op", options=opts)
total_flops = flops.total_float_ops

print("Model FLOPS:", total_flops, "==", round(total_flops/1e9,3), "GFLOPS")