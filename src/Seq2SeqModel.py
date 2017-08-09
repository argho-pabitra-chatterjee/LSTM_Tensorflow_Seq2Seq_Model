## Implementing Sequencing Modelling
## will try to make a bi-directional encoder and decoder using tensor flow
## So we can talk about the future and the past both, via model using pattern recognition 
import tensorflow as tf # this is for ML
import helpers as helpers

# clearing cache in case of multiple graphs 
tf.reset_default_graph()
session = tf.InteractiveSession()

PAD = 0
EOS = 1

## here we are using small sequences of 10 length vector
vocab_size = 10
input_embedding_size = 20

ecoder_hidden_units = 20
decoder_hidden_units = ecoder_hidden_units * 2


# placeholders
encoder_in = tf.placeholder(dtype = tf.int32, shape=(None,None), name='encoder_input')
encoder_inputs_length = tf.placeholder(dtype = tf.int32, shape = (None,), name = 'encoder_input_length')
decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None,None), name='decoder_targets')


# defining embeddings
embeddings = tf.Variable(tf.random_uniform([vocab_size,input_embedding_size],-1.0,1), dtype = tf.float32)
# Randomly initialized the embedding matrix to fit the input sequence
# we feed the enbedding matrix into the designed encoder
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_in)


## Designing the Encoder
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple

## here each neuron is an LSTM
encoder_cell = LSTMCell(ecoder_hidden_units)


((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float64, time_major=True)
    )
  
  
## bidirectional step
encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

## TF tuples used by LSTM cells for state size, zero_state and output state.
encoder_final_state = LSTMStateTuple(c = encoder_final_state_c, h = encoder_final_state_h)
## c is the internal state of the cell ; h is the output value
## we combine internal state and output value and then feed this to the decoder



## Designing the Decoder
decoder_cell = LSTMCell(decoder_hidden_units)       # the number of hidden units is double that of the input
# we feed our data into batches
encodr_maxtime, batch_size = tf.unstack(tf.shape(encoder_in))
# since models do not know end of sentences, we add +3 to add EOS
decoder_lengths = encoder_inputs_length + 3



## defining weights 
W = tf.Variable(tf.random_uniform([decoder_hidden_units],vocab_size,-1,1),dtype=tf.float32)
## defining biases
b = tf.Variable(tf.zeros([vocab_size]),dtype = tf.float32)


#create padded inputs for the decoder from the word embeddings
#were telling the program to test a condition, and trigger an error if the condition is false.
assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice) 
###




#manually specifying loop function through time - to get initial cell state and input to RNN
#normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

#we define and return these values, no operations occur here
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded
    #last time steps cell state
    initial_cell_state = encoder_final_state
    #none
    initial_cell_output = None
    #none
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)
    
    


#attention mechanism --choose which previously generated token to pass as input in the next timestep
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    
    def get_next_in():
        #dot product between previous ouput and weights, then + biases
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        #Logits simply means that the function operates on the unscaled output of 
        #earlier layers and that the relative scale to understand the units is linear. 
        #It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities 
        #(you might have an input of 5).
        #prediction value at current time step
        
        #Returns the index with the largest value across axes of a tensor.
        prediction = tf.argmax(output_logits, axis=1)
        #embed prediction for the next input
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    
    ele_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended

    
    
    #Computes the "logical and" of elements across dimensions of a tensor.
    finished = tf.reduce_all(ele_finished) # -> boolean scalar
    #Return either fn1() or fn2() based on the boolean predicate pred.
    in_ = tf.cond(finished, lambda: pad_step_embedded, get_next_in)
    
    #set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (ele_finished, 
            in_,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

#Creates an RNN specified by RNNCell cell and loop function loop_fn.
#This function is a more primitive version of dynamic_rnn that provides more direct access to the 
#inputs each iteration. It also provides more control over when to start and finish reading the sequence, 
#and what to emit for the output.
#ta = tensor array
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()



#to convert output to human readable prediction
#we will reshape output tensor

#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#reduces dimensionality
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
#flettened output tensor
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
#pass flattened tensor through decoder
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
#prediction vals
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))





#cross entropy loss
#one hot encode the target values so we don't rank just differentiate
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

#loss function
loss = tf.reduce_mean(stepwise_cross_entropy)
#train it 
train_op = tf.train.AdamOptimizer().minimize(loss)


session.run(tf.global_variables_initializer())



batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)
    
    
def next_feed():
    batch = next(batches)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
    )
    return {
        encoder_in: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }
loss_track = []


max_batches = 3001
batches_in_epoch = 1000
decoder_pred = tf.argmax(decoder_logits, 2)

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = session.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(session.run(loss, fd)))
            predict_ = session.run(decoder_pred, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_in].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()

except KeyboardInterrupt:
    print('training interrupted')
    


import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))