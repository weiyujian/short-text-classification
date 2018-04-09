#get last output of lstm for classification
#if use bidirectional_dynamic_rnn,for forward cell the last output is in the right hand 
#and for backward cell the last output is in the left hand
"""
rnn method1 : get the output by index
"""
import numpy as np
import tensorflow as tf

# Create input data
X = np.random.randn(2, 10, 8)
n_hidden = 2 
# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]
 
cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True)
 
outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

batch_size = tf.shape(X)[0]
seq_max_len = max(X_lengths)
inx = tf.range(0.batch_size) * seq_max_len + X_lengths - 1
outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]),inx)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	o = sess.run(outputs)
	s = sess.run(last_states)
	print o
	print s

"""
rnn method2 : use the last states as the output directly
"""
import numpy as np
import tensorflow as tf

# Create input data
X = np.random.randn(2, 10, 8)
n_hidden = 2 
# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]
 
cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True)
 
outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)
	
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	o = sess.run(outputs)
	s = sess.run(last_states)
	print o
	print s

	
"""
bi-rnn method1 : get the output by index
"""
import numpy as np
import tensorflow as tf

# Create input data
X = np.random.randn(2, 10, 8)
n_hidden = 2 
# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]
 
fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True)
bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True)
 
outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=fw_cell,
    cell_bw=bw_cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

batch_size = tf.shape(X)[0]
seq_max_len = max(X_lengths)
index_start = tf.range(0.batch_size) * seq_max_len
index_end = tf.range(0.batch_size) * seq_max_len + X_lengths - 1
output_fw = tf.gather(tf.reshape(outputs, [-1, n_hidden]),index_end)
output_bw = tf.gather(tf.reshape(outputs, [-1, n_hidden]),index_start)
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	o_fw = sess.run(outputs_fw)
	o_bw = sess.run(outputs_bw)
	print o_fw
	print o_bw
	
"""
bi-rnn method2 : use the last states as the output directly
"""
import numpy as np
import tensorflow as tf

# Create input data
X = np.random.randn(2, 10, 8)
n_hidden = 2 
# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]
 
fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True)
bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True)
 
outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=fw_cell,
    cell_bw=bw_cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

output_fw,output_bw = outputs
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	o_fw = sess.run(outputs_fw)
	o_bw = sess.run(outputs_bw)
	print o_fw
	print o_bw
