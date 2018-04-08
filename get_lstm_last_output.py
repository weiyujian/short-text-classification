#get last output of lstm for classification
#if use bidirectional_dynamic_rnn,for forward cell the last output is in the right hand 
#and for backward cell the last output is in the left hand
import numpy as np
import tensorflow as tf

# Create input data
X = np.random.randn(2, 10, 8)
 
# The second example is of length 6 
X[1,6:] = 0
X_lengths = [10, 6]
 
cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
 
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