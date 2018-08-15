#### We use argparse for processing command line arguments, random for shuffling our data, sys for flushing output, and numpy for handling vectors of data.
# Tensorflow Implementation
import argparse
import random
import sys

import numpy as np

#### Typically, we would make many of these constants command line arguments and tune using the development set. For simplicity, I have fixed their values here to match Jiang, Liang and Zhang (CoLing 2018).
PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 100 # DIM_EMBEDDING - number of dimensions in our word embeddings.
LSTM_HIDDEN = 100 # LSTM_HIDDEN - number of dimensions in the hidden vectors for the LSTM. Based on NCRFpp (200 in the paper, but 100 per direction in code) 
BATCH_SIZE = 10 # BATCH_SIZE - number of examples considered in each model update.
LEARNING_RATE = 0.015 # LEARNING_RATE - adjusts how rapidly model parameters change by rescaling the gradient vector.
LEARNING_DECAY_RATE = 0.05 # LEARNING_DECAY_RATE - part of a rescaling of the learning rate after each pass through the data.
EPOCHS = 100 # EPOCHS - number of passes through the data in training.
KEEP_PROB = 0.5 # KEEP_PROB - probability of keeping a value when applying dropout.
GLOVE = "../data/glove.6B.100d.txt" # GLOVE - location of glove vectors.
# WEIGHT_DECAY = 1e-8 Not used, see note at the bottom of the page

#### Tensorflow library import.
import tensorflow as tf

####
# Data reading
def read_data(filename):
    #### We are expecting a minor variation on the raw Penn Treebank data, with one line per sentence, tokens separated by spaces, and the tag for each token placed next to its word (the | works as a separator as it does not appear as a token).
    """Example input:
    Pierre|NNP Vinken|NNP ,|, 61|CD years|NNS old|JJ
    """
    content = []
    with open(filename) as data_src:
        for line in data_src:
            t_p = [w.split("|") for w in line.strip().split()]
            tokens = [v[0] for v in t_p]
            tags = [v[1] for v in t_p]
            content.append((tokens, tags))
    return content

def simplify_token(token):
    chars = []
    for char in token:
        #### Reduce sparsity by replacing all digits with 0.
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

def main():
    #### For the purpose of this example we only have arguments for locations of the data.
    parser = argparse.ArgumentParser(description='POS tagger.')
    parser.add_argument('training_data')
    parser.add_argument('dev_data')
    args = parser.parse_args()

    train = read_data(args.training_data)
    dev = read_data(args.dev_data)

    #### These indices map from strings to integers, which we apply to the input for our model. UNK is added to our mapping so that there is a vector we can use when we encounter unknown words. The special PAD symbol is used in PyTorch and Tensorflow as part of shaping the data in a batch to be a consistent size. It is not needed for DyNet, but kept for consistency.
    # Make indices
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
    #### The '+ dev' may seem like an error, but is done here for convenience. It means in the next section we will retain the GloVe embeddings that appear in dev but not train. They won't be updated during training, so it does not mean we are getting information we shouldn't. In practise I would simply keep all the GloVe embeddings to avoid any potential incorrect use of the evaluation data.
    for tokens, tags in train + dev:
        for token in tokens:
            token = simplify_token(token)
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
                id_to_token.append(token)
        for tag in tags:
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
                id_to_tag.append(tag)
    NWORDS = len(token_to_id)
    NTAGS = len(tag_to_id)

    # Load pre-trained GloVe vectors
    #### I am assuming these are 100-dimensional GloVe embeddings in their standard format.
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
    #### We need the word vectors as a list to initialise the embeddings. Each entry in the list corresponds to the token with that index.
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING)
    for word in id_to_token:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            #### For words that do not appear in GloVe we generate a random vector (note, the choice of scale here is important and we follow Jiang, Liang and Zhang (CoLing 2018).
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)

    #### The most significant difference between the frameworks is how the model parameters and their execution is defined. In DyNet we define parameters here and then define computation as needed. In PyTorch we use a class with the parameters defined in the constructor and the computation defined in the forward() method. In Tensorflow we define both parameters and computation here.
    # Model creation
    ####
    #### This line creates a new graph and makes it the default graph for operations to be registered to. It is not necessary here because we only have one graph, but is considered good practise (more discussion on <a href="https://stackoverflow.com/questions/39614938/why-do-we-need-tensorflow-tf-graph">Stackoverflow</a>.
    with tf.Graph().as_default():
        #### Placeholders are inputs/values that will be fed into the network each time it is run. We define their type, name, and shape (constant, 1D vector, 2D vector, etc). This includes what we normally think of as inputs (e.g. the tokens) as well as parameters we want to change at run time (e.g. the learning rate).
        # Define inputs
        e_input = tf.placeholder(tf.int32, [None, None], name='input')
        e_lengths = tf.placeholder(tf.int32, [None], name='lengths')
        e_mask = tf.placeholder(tf.int32, [None, None], name='mask')
        e_gold_output = tf.placeholder(tf.int32, [None, None],
                name='gold_output')
        e_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        e_learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Define word embedding
        #### The embedding matrix is a variable (so they can shift in training), initialized with the vectors defined above.
        glove_init = tf.constant_initializer(np.array(pretrained_list))
        e_embedding = tf.get_variable("embedding", [NWORDS, DIM_EMBEDDING],
                initializer=glove_init)
        e_embed = tf.nn.embedding_lookup(e_embedding, e_input)

        # Define LSTM cells
        #### We create an LSTM cell, then wrap it in a class that applies dropout to the input and output.
        e_cell_f = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN)
        e_cell_f = tf.contrib.rnn.DropoutWrapper(e_cell_f,
                input_keep_prob=e_keep_prob, output_keep_prob=e_keep_prob)
        # Recurrent dropout options
        #### We are not using recurrent dropout, but it is a common enough feature of networks that it's good to see how it is done.
        #        variational_recurrent=True, dtype=tf.float32,
        #        input_size=DIM_EMBEDDING)
        #### Similarly, multi-layer networks are a common use case. In Tensorflow, we would wrap a list of cells with a MultiRNNCell.
        # Multi-layer cell creation
        # e_cell_f = tf.contrib.rnn.MultiRNNCell([e_cell_f])
        #### We are making a bidirectional network, so we need another cell for the reverse direction.
        e_cell_b = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN)
        e_cell_b = tf.contrib.rnn.DropoutWrapper(e_cell_b,
                input_keep_prob=e_keep_prob, output_keep_prob=e_keep_prob)
        #### To use the cells we create a dynamic RNN. The 'dynamic' aspect means we can feed in the lengths of input sequences not counting padding and it will stop early.
        e_initial_state_f = e_cell_f.zero_state(BATCH_SIZE, dtype=tf.float32)
        e_initial_state_b = e_cell_f.zero_state(BATCH_SIZE, dtype=tf.float32)
        e_lstm_outputs, e_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=e_cell_f, cell_bw=e_cell_b, inputs=e_embed,
                initial_state_fw=e_initial_state_f,
                initial_state_bw=e_initial_state_b,
                sequence_length=e_lengths, dtype=tf.float32)
        e_lstm_outputs_merged = tf.concat(e_lstm_outputs, 2)

        # Define output layer
        #### Matrix multiply to get scores for each class.
        e_predictions = tf.contrib.layers.fully_connected(e_lstm_outputs_merged,
                NTAGS, activation_fn=None)
        # Define loss and update
        #### Cross-entropy loss. The reduction flag is crucial (the default is to average over the sequence). The weights flag accounts for padding that makes all of the sequences the same length.
        e_loss = tf.losses.sparse_softmax_cross_entropy(e_gold_output,
                e_predictions, weights=e_mask,
                reduction=tf.losses.Reduction.SUM)
        e_train = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(e_loss)
        # Update with gradient clipping
        #### If we wanted to do gradient clipping we would need to do the update in a few steps, first calculating the gradient, then modifying it before applying it.
        # e_optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # e_gradients = e_optimiser.compute_gradients(e_loss)
        # e_clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
        #         for grad, var in e_gradients]
        # e_train = e_optimiser.apply_gradients(e_gradients)

        # Define output
        e_auto_output = tf.argmax(e_predictions, 2, output_type=tf.int32)

        # Do training
        #### Configure the system environment. By default Tensorflow uses all available GPUs and RAM. These lines limit the number of GPUs used and the amount of RAM. To limit which GPUs are used, set the environment variable CUDA_VISIBLE_DEVICES (e.g. "export CUDA_VISIBLE_DEVICES=0,1").
        config = tf.ConfigProto(
            device_count = {'GPU': 0},
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        )
        #### A session runs the graph. We use a 'with' block to ensure it is closed, which frees various resources.
        with tf.Session(config=config) as sess:
            #### Run executes operations, in this case initializing the variables.
            sess.run(tf.global_variables_initializer())

            #### To make the code match across the three versions, we group together some framework specific values needed when doing a pass over the data.
            expressions = [
                e_auto_output, e_gold_output, e_input, e_keep_prob, e_lengths,
                e_loss, e_train, e_mask, e_learning_rate, sess
            ]
            #### Main training loop, in which we shuffle the data, set the learning rate, do one complete pass over the training data, then evaluate on the development data.
            for epoch in range(EPOCHS):
                random.shuffle(train)

                ####
                # Determine the current learning rate
                current_lr = LEARNING_RATE / (1+ LEARNING_DECAY_RATE * epoch)

                #### Training pass.
                loss, tacc = do_pass(train, token_to_id, tag_to_id, expressions,
                        True, current_lr)
                #### Dev pass.
                _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions,
                        False)
                print("{} loss {} t-acc {} d-acc {}".format(epoch, loss, tacc,
                    dacc))

            #### The syntax varies, but in all three cases either saving or loading the parameters of a model must be done after the model is defined.
            # Save model
            saver = tf.train.Saver()
            saver.save(sess, "./tagger.tf.model")

            # Load model
            saver.restore(sess, "./tagger.tf.model")

            # Evaluation pass.
            _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions,
                    False)
            print("Test Accuracy: {:.3f}".format(test_acc))

#### Inference (the same function for train and test).
def do_pass(data, token_to_id, tag_to_id, expressions, train, lr=0.0):
    e_auto_output, e_gold_output, e_input, e_keep_prob, e_lengths, e_loss, \
            e_train, e_mask, e_learning_rate, session = expressions

    # Loop over batches
    loss = 0
    match = 0
    total = 0
    for start in range(0, len(data), BATCH_SIZE):
        #### Form the batch and order it based on length (important for efficient processing in PyTorch).
        batch = data[start : start + BATCH_SIZE]
        batch.sort(key = lambda x: -len(x[0]))
        #### Log partial results so we can conveniently check progress.
        if start % 4000 == 0 and start > 0:
            print(loss, match / total)
            sys.stdout.flush()

        ####
        # Add empty sentences to fill the batch
        #### We add empty sentences because Tensorflow requires every batch to be the same size.
        batch += [([], []) for _ in range(BATCH_SIZE - len(batch))]
        # Prepare inputs
        #### We do this here for convenience and to have greater alignment between implementations, but in practise it would be best to do this once in pre-processing.
        max_length = len(batch[0][0])
        input_array = np.zeros([len(batch), max_length])
        output_array = np.zeros([len(batch), max_length])
        lengths = np.array([len(v[0]) for v in batch])
        mask = np.zeros([len(batch), max_length])
        #### Convert tokens and tags from strings to numbers using the indices.
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            #### Fill the arrays, leaving the remaining values as zero (our padding value).
            input_array[n, :len(tokens)] = token_ids
            output_array[n, :len(tags)] = tag_ids
            mask[n, :len(tokens)] = np.ones([len(tokens)])
        #### We can't change the computation graph to disable dropout when not training, so we just change the keep probability.
        cur_keep_prob = KEEP_PROB if train else 1.0
        #### This dictionary contains values for all of the placeholders we defined.
        feed = {
                e_input: input_array,
                e_gold_output: output_array,
                e_mask: mask,
                e_keep_prob: cur_keep_prob,
                e_lengths: lengths,
                e_learning_rate: lr
        }

        # Define the computations needed
        todo = [e_auto_output]
        #### If we are not training we do not need to compute a loss and we do not want to do the update.
        if train:
            todo.append(e_loss)
            todo.append(e_train)
        # Run computations
        outcomes = session.run(todo, feed_dict=feed)
        # Get outputs
        predicted = outcomes[0]
        if train:
            #### We do not request the e_train value because its work is done - it performed the update during its computation.
            loss += outcomes[1]

        ####
        # Update the number of correct tags and total tags
        for (_, g), a in zip(batch, predicted):
            total += len(g)
            for gt, at in zip(g, a):
                gt = tag_to_id[gt]
                if gt == at:
                    match += 1

    return loss, match / total

if __name__ == '__main__':
    main()
