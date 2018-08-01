#!/usr/bin/env python3

import argparse
import random
import sys

import numpy as np
import tensorflow as tf

PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 100
LSTM_SIZES = [100] # based on NCRFpp (200 in the paper, but 100 per direction in code)
BATCH_SIZE = 10
LEARNING_RATE = 0.015
LEARNING_DECAY_RATE = 0.05
EPOCHS = 100
KEEP_PROB = 0.5
GLOVE = "../data/glove.6B.100d.txt"
WEIGHT_DECAY = 1e-8 # TODO apply as L2 regularization

def read_data(filename):
    """Example input:
    Pierre|NNP Vinken|NNP ,|, 61|CD years|NNS old|JJ ,|, will|MD join|VB the|DT board|NN as|IN a|DT nonexecutive|JJ director|NN Nov.|NNP 29|CD .|.
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
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

def main():
    parser = argparse.ArgumentParser(description='Tensorflow tagger.')
    parser.add_argument('training_data')
    parser.add_argument('dev_data')
    args = parser.parse_args()

    # Read data
    train = read_data(args.training_data)
    dev = read_data(args.dev_data)

    # Make indices
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
    for tokens, tags in train:
        for token in tokens:
            token = simplify_token(token)
            if token not in token_to_id:
                token_to_id[token] = len(id_to_token)
                id_to_token.append(token)
        for tag in tags:
            if tag not in tag_to_id:
                tag_to_id[tag] = len(id_to_tag)
                id_to_tag.append(tag)

    # Load pre-trained vectors
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING) # From Jiang, Liang and Zhang
    for word in id_to_token:
        if word in pretrained:
            pretrained_list.append(np.array(pretrained[word]))
        elif word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            pretrained_list.append(np.random.uniform(-scale, scale, [DIM_EMBEDDING]))

    # Construct computation graph
    with tf.Graph().as_default():
        e_input = tf.placeholder(tf.int32, [None, None], name='input')
        e_lengths = tf.placeholder(tf.int32, [None], name='lengths')
        e_mask = tf.placeholder(tf.int32, [None, None], name='mask')
        e_gold_output = tf.placeholder(tf.int32, [None, None], name='gold_output')
        e_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        e_learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        glove_init = tf.constant_initializer(np.array(pretrained_list))
        e_embedding = tf.get_variable("embedding", [len(id_to_token), DIM_EMBEDDING], initializer=glove_init)
        e_embed = tf.nn.embedding_lookup(e_embedding, e_input)

        e_raw_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZES[0])
        e_cell = tf.contrib.rnn.DropoutWrapper(e_raw_cell,
                input_keep_prob=e_keep_prob, output_keep_prob=e_keep_prob)
###        # Recurrent dropout
###                variational_recurrent=True, dtype=tf.float32,
###                input_size=DIM_EMBEDDING)
###        # Creating a stack of layers
###        e_cell = tf.contrib.rnn.MultiRNNCell(e_dropped_lstm)

        initial_state = e_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        e_lstm_outputs, e_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=e_cell, cell_bw=e_cell, inputs=e_embed,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                sequence_length=e_lengths, dtype=tf.float32)
        e_lstm_outputs_merged = tf.concat(e_lstm_outputs, 2)

        e_predictions = tf.contrib.layers.fully_connected(e_lstm_outputs_merged,
                len(id_to_tag), activation_fn=None)
        e_loss = tf.losses.sparse_softmax_cross_entropy(e_gold_output,
                e_predictions, weights=e_mask, reduction=tf.losses.Reduction.SUM)
        # One step option
        e_train = tf.train.GradientDescentOptimizer(e_learning_rate).minimize(e_loss)
###        # Multi-step, so that gradient clipping can be applied
###        e_optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE)
###        e_gradients = e_optimiser.compute_gradients(e_loss)
###        e_clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in e_gradients]
###        e_train = e_optimiser.apply_gradients(e_gradients)

        e_auto_output = tf.argmax(e_predictions, 2, output_type=tf.int32)

        expressions = [
            e_auto_output,
            e_gold_output,
            e_input,
            e_keep_prob,
            e_lengths,
            e_loss,
            e_train,
            e_mask,
            e_learning_rate
        ]
        
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
            for epoch_no in range(EPOCHS):
                random.shuffle(train)
                current_lr = LEARNING_RATE / (1 + LEARNING_DECAY_RATE * epoch_no)

                train_loss, train_total, train_match = do_pass(train, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, sess, True, current_lr)
                _, dev_total, dev_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, sess)

                print("epoch {} t-loss {} t-acc {} d-acc {}".format(epoch_no, train_loss, train_match / train_total, dev_match / dev_total))
 
        # Save model
###            saver.save(sess, "checkpoints/tagger.ckpt")
        # Reload model
###            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

            _, test_total, test_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, sess)
            print("Test Accuracy: {:.3f}".format(test_match / test_total))

def do_pass(data, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, session, train=False, lr=0.0):
    e_auto_output, e_gold_output, e_input, e_keep_prob, e_lengths, e_loss, e_train, e_mask, e_learning_rate = expressions

    loss = 0
    match = 0
    total = 0
    start = 0
    while start < len(data):
        batch = data[start : start + BATCH_SIZE]
        batch.sort(key = lambda x: -len(x[0]))
        start += BATCH_SIZE
        if start % 4000 == 0:
            print(loss, match / total)

        max_length = len(batch[0][0])
        batch += [([], []) for _ in range(BATCH_SIZE - len(batch))] # Add empty sentences to fill in batch
        input_array = np.zeros([len(batch), max_length])
        output_array = np.zeros([len(batch), max_length])
        lengths = np.array([len(v[0]) for v in batch])
        mask = np.zeros([len(batch), max_length])
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), token_to_id[UNK]) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]
            input_array[n, :len(tokens)] = token_ids
            output_array[n, :len(tokens)] = tag_ids
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

        for (_, g), a in zip(batch, predicted):
            total += len(g)
            for gt, at in zip(g, a):
                gt = tag_to_id[gt]
                if gt == at:
                    match += 1

    return loss, total, match

if __name__ == '__main__':
    main()
