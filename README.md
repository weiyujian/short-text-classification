# 使用rnn进行短文本分类

	使用rnn进行分类的时候，我们需要固定输入文本的大小input_size，但是我们会遇到这样一个问题：文本的实际是变长的，我们需要通过截断和padding的方法使得输入文本的长度一致，才能调用tensorflow的api。
	因此，指定sequence_length参数有两个好处：1.提高计算效率2.使得输出结果正确
	试想下，如果不传入这个参数，假设一个文本长13，另一个长28，我们事先需要把13的文本补零到28长度，这样就会遇到问题，对于13的文本，当rnn走到第13步的时候，其实已经结束了，后面的output都会被置为0，而对于28文本，rnn需要走到第28步才结束。如果不传入sequence_length=[13,28]这个参数的话，tensorflow会一直计算隐层状态直至28步，这样计算出来的隐层状态是包含padding元素的，得到的结果是不对的。相反，当我们传入sequence_length参数的话，当tensorflow计算到第13步状态时，就会停止计算了，然后就copy这个隐层状态一直到第28步结束，而output的话会从第13步开始一直到28步都置为0
	
	通过上面的分析我们知道，对于变长文本，我们可以通过调用dynamic_rnn API来跳过padding部分的计算，从而1.减少计算量2.得到正确结果
	对于上面的例子来说，调用dynamic_rnn后，对于第一个样本，output从13步开始output就置为0了，而state从13步开始始终保持第13步的状态
	对于变长文本，我们要获得的肯定是实际的输出，而不是没有用的padding输出，有两种方法可以获得实际的输出
	1.参考 https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
	假设我们已经得到了rnn的输出outputs[None,seq_max_len,n_hidden]以及传进来的每个样本实际的length：seqlen
	cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=2,state_is_tuple=True)
	
	outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=seq_max_len,
    inputs=X)
	
	batch_size = tf.shape(outputs)[0]
	index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
	outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
	2.根据API的定义来获得实际输出
	因为rnn的输出其实就是隐层状态的h（LSTM而言，rnn和GRU的输出与隐层状态是一样的），因此last_states中的h状态就是我们所需的实际的最后的output：
	cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=2,state_is_tuple=True)
	
	outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=seq_max_len,
    inputs=X)

	outputs = last_states.h


refer:http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
