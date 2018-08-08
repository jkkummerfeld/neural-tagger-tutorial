#### Imports
import argparse
import random
import sys

import numpy as np

#### Constants
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
WEIGHT_DECAY = 1e-8 # WEIGHT_DECAY - part of a rescaling of weights when an update occurs.  TODO apply as L2 regularization

#### Tensorflow specfic import
import tensorflow as tf

#### Reading the data
def read_data(filename):
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

#### Replace all digits with 0 to decrease sparsity.
def simplify_token(token):
    chars = []
    for char in token:
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

def main():
    #### Read arguments
    parser = argparse.ArgumentParser(description='POS tagger.')
    parser.add_argument('training_data')
    parser.add_argument('dev_data')
    args = parser.parse_args()

    #### Read data (see function above)
    train = read_data(args.training_data)
    dev = read_data(args.dev_data)

    #### Make indices
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
    for tokens, tags in train + dev: # dev is necessary here to get the GloVe embeddings for words in dev but not train loaded. They will not be updated during training.
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

    #### Load pre-trained vectors
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
    #### We need the word vectors as a list to initialise the embeddings. Each entry in the list corresponds to the token with that index.
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING) # From Jiang, Liang and Zhang
    for word in id_to_token:
        if word.lower() in pretrained: # applying lower() here because all GloVe vectors are for lowercase words
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            #### For words that do not appear in GloVe we generate a random vector (note, the choice of scale here is important).
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)

    #### Construct computation graph
    with tf.Graph().as_default():
        e_input = tf.placeholder(tf.int32, [None, None], name='input')
        e_lengths = tf.placeholder(tf.int32, [None], name='lengths')
        e_mask = tf.placeholder(tf.int32, [None, None], name='mask')
        e_gold_output = tf.placeholder(tf.int32, [None, None],
                name='gold_output')
        e_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        e_learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        glove_init = tf.constant_initializer(np.array(pretrained_list))
        e_embedding = tf.get_variable("embedding", [NWORDS, DIM_EMBEDDING],
                initializer=glove_init)
        e_embed = tf.nn.embedding_lookup(e_embedding, e_input)

        e_raw_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN)
        e_cell = tf.contrib.rnn.DropoutWrapper(e_raw_cell,
                input_keep_prob=e_keep_prob, output_keep_prob=e_keep_prob)
        #### Recurrent dropout
        #        variational_recurrent=True, dtype=tf.float32,
        #        input_size=DIM_EMBEDDING)
        #### Creating a stack of layers
        # e_cell = tf.contrib.rnn.MultiRNNCell(e_dropped_lstm)

        initial_state = e_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        e_lstm_outputs, e_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=e_cell, cell_bw=e_cell, inputs=e_embed,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                sequence_length=e_lengths, dtype=tf.float32)
        e_lstm_outputs_merged = tf.concat(e_lstm_outputs, 2)

        e_predictions = tf.contrib.layers.fully_connected(e_lstm_outputs_merged,
                NTAGS, activation_fn=None)
        e_loss = tf.losses.sparse_softmax_cross_entropy(e_gold_output,
                e_predictions, weights=e_mask,
                reduction=tf.losses.Reduction.SUM)
        #### One step option
        e_train = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(e_loss)
        #### Multi-step, so that gradient clipping can be applied
        # e_optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # e_gradients = e_optimiser.compute_gradients(e_loss)
        # e_clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in e_gradients]
        # e_train = e_optimiser.apply_gradients(e_gradients)

        e_auto_output = tf.argmax(e_predictions, 2, output_type=tf.int32)

        # Use computation graph
        saver = tf.train.Saver()
        config = tf.ConfigProto(
            # Set CUDA_VISIBLE_DEVICES to control which GPUs are visible
            # Then adjust these:
            device_count = {'GPU': 0},
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        )
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            expressions = [
                e_auto_output,
                e_gold_output,
                e_input,
                e_keep_prob,
                e_lengths,
                e_loss,
                e_train,
                e_mask,
                e_learning_rate,
                sess
            ]
        
            #### Main training loop, in which we shuffle the data, set the learning rate, do one complete pass over the training data, then evaluate on the development data.
            for epoch in range(EPOCHS):
                random.shuffle(train)

                #### Determine the current learning rate
                current_lr = LEARNING_RATE / (1+ LEARNING_DECAY_RATE * epoch)

                #### Training pass
                loss, tacc = do_pass(train, token_to_id, tag_to_id,
                        expressions, True, current_lr)
                #### Dev pass
                _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions,
                        False)
                print("{} loss {} t-acc {} d-acc {}".format(epoch, loss, tacc, dacc))

            #### Save model
            # saver.save(sess, "checkpoints/tagger.ckpt")

            #### Reload model
            # saver.restore(sess, tf.train.latest_checkpoint(model_dir))

            #### Evaluation
            _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions,
                    False)
            print("Test Accuracy: {:.3f}".format(test_acc))

#### Inference (the same function for train and test)
def do_pass(data, token_to_id, tag_to_id, expressions, train, lr=0.0):
    #### Get expressions
    e_auto_output, e_gold_output, e_input, e_keep_prob, e_lengths, e_loss, \
            e_train, e_mask, e_learning_rate, session = expressions

    #### Loop over batches, tracking the start of the batch in the data
    loss = 0
    match = 0
    total = 0
    start = 0
    while start < len(data):
        #### Form the batch and order it based on length (not necessary for DyNet, but important for efficient processing in PyTorch).
        batch = data[start : start + BATCH_SIZE]
        batch.sort(key = lambda x: -len(x[0]))
        start += BATCH_SIZE
        #### Log partial results so we can conveniently check progress
        if start % 4000 == 0:
            print(loss, match / total)
            sys.stdout.flush()

        # Tensorflow specific
        max_length = len(batch[0][0])
        batch += [([], []) for _ in range(BATCH_SIZE - len(batch))] # Add empty sentences to fill in batch
        input_array = np.zeros([len(batch), max_length])
        output_array = np.zeros([len(batch), max_length])
        lengths = np.array([len(v[0]) for v in batch])
        mask = np.zeros([len(batch), max_length])
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]
            input_array[n, :len(tokens)] = token_ids
            output_array[n, :len(tags)] = tag_ids
            mask[n, :len(tokens)] = np.ones([len(tokens)])
        cur_keep_prob = KEEP_PROB
        if not train:
            cur_keep_prob = 1.0

        feed = {
                e_input: input_array,
                e_gold_output: output_array,
                e_mask: mask,
                e_keep_prob: cur_keep_prob,
                e_lengths: lengths,
                e_learning_rate: lr
        }
        todo = [e_auto_output]
        if train:
            todo.append(e_loss)
            todo.append(e_train)
        outcomes = session.run(todo, feed_dict=feed)
        predicted = outcomes[0]
        if train:
            loss += outcomes[1]

        #### Update the number of correct tags and total tags
        for (_, g), a in zip(batch, predicted):
            total += len(g)
            for gt, at in zip(g, a):
                gt = tag_to_id[gt]
                if gt == at:
                    match += 1

    return loss, total / match

if __name__ == '__main__':
    main()
