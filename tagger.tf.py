#!/usr/bin/env python3

import argparse
import random

import numpy as np
import tensorflow as tf

PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 100
LSTM_SIZES = [100] # based on NCRFpp (200 in the paper, but 100 per direction in code)
BATCH_SIZE = 10
LEARNING_RATE = 0.015
LEARNING_DECAY_RATE = 0.05 # TODO apply
EPOCHS = 100
KEEP_PROB = 0.5

# TODO: L2 regularization 1e-8

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
    token_to_id = {PAD: 0, UNK:1}
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

    # TODO: Read GloVe

    with tf.Graph().as_default():
        # Construct computation graph
        e_input = tf.placeholder(tf.int32, [None, None], name='input')
        e_lengths = tf.placeholder(tf.int32, [None], name='lengths')
        e_mask = tf.placeholder(tf.int32, [None, None], name='mask')
        e_gold_output = tf.placeholder(tf.int32, [None, None], name='gold_output')
        e_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        e_embedding = tf.get_variable("embedding", [len(id_to_token), DIM_EMBEDDING])
        e_embed = tf.nn.embedding_lookup(e_embedding, e_input)

###        e_lstm = [tf.contrib.rnn.BasicLSTMCell(size) for size in LSTM_SIZES]
###        e_dropped_lstm = [tf.contrib.rnn.DropoutWrapper(l, output_keep_prob=e_keep_prob) for l in e_lstm]
###        e_cell = tf.contrib.rnn.MultiRNNCell(e_dropped_lstm)

        e_raw_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZES[0])
        e_cell = tf.contrib.rnn.DropoutWrapper(e_raw_cell,
                input_keep_prob=e_keep_prob, output_keep_prob=e_keep_prob,
                variational_recurrent=True, dtype=tf.float32,
                input_size=DIM_EMBEDDING)

        # TODO tf.nn.bidirectional_dynamic_rnn
        initial_state = e_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
        e_lstm_outputs, e_final_state = tf.nn.dynamic_rnn(e_cell, e_embed,
                initial_state=initial_state,
                sequence_length=e_lengths, dtype=tf.float32)

        e_predictions = tf.contrib.layers.fully_connected(e_lstm_outputs,
                len(id_to_tag), activation_fn=None)
        e_loss = tf.losses.sparse_softmax_cross_entropy(e_gold_output,
                e_predictions, weights=e_mask)
###        e_train = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(e_loss)
        e_optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        e_gradients = e_optimiser.compute_gradients(e_loss)
        e_clipped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in e_gradients]
        e_train = e_optimiser.apply_gradients(e_clipped_gradients)

        e_auto_output = tf.argmax(e_predictions, 2, output_type=tf.int32)

        expressions = [
            e_auto_output,
            e_gold_output,
            e_input,
            e_keep_prob,
            e_lengths,
            e_loss,
            e_train,
            e_mask
        ]
        
        # Use computation graph
        saver = tf.train.Saver()

        config = tf.ConfigProto(
                # Set CUDA_VISIBLE_DEVICES to control which GPUs are visible
                # Then adjust these:
            device_count = {'GPU': 0},
###            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        )
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_no in range(EPOCHS):
                random.shuffle(train)
                train_loss, train_total, train_match = do_pass(train, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, sess, True)
                _, dev_total, dev_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, sess)

                print("epoch {} t-loss {} t-acc {} d-acc {}".format(epoch_no, train_loss, train_match / train_total, dev_match / dev_total))
 
        # Save model
###            saver.save(sess, "checkpoints/tagger.ckpt")
        # Reload model
###            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

            _, test_total, test_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, sess)
            print("Test Accuracy: {:.3f}".format(test_match / test_total))

def do_pass(data, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, session, train=False):
    # TODO: Fix so it doesn't miss the end (anything that is not a full batch)
    e_auto_output, e_gold_output, e_input, e_keep_prob, e_lengths, e_loss, e_train, e_mask = expressions
    cur_keep_prob = KEEP_PROB
    if not train:
        cur_keep_prob = 1.0
    loss = 0
    match = 0
    total = 0
    start = 0
    while start + BATCH_SIZE < len(data):
        batch = data[start : start + BATCH_SIZE]
        start += BATCH_SIZE
        max_length = max(len(v[0]) for v in batch)
        x = []
        y = []
        lengths = []
        mask = []
        for tokens, tags in batch:
            lengths.append(len(tokens))
            mask.append(np.array(
                [1.0] * len(tokens) +
                [0.0] * (max_length - len(tokens)) ))

            tokens += [PAD] * (max_length - len(tokens))
            tags += [PAD] * (max_length - len(tags))

            tokens = [token_to_id.get(simplify_token(t), token_to_id[UNK]) for t in tokens]
            tags = [tag_to_id[t] for t in tags]

            x.append(np.array(tokens))
            y.append(np.array(tags))

        feed = {
                e_input: np.array(x),
                e_gold_output: np.array(y),
                e_mask: np.array(mask),
                e_keep_prob: cur_keep_prob,
                e_lengths: np.array(lengths)
        }
        todo = [e_gold_output, e_auto_output]
        if train:
            todo.append(e_loss)
            todo.append(e_train)
        outcomes = session.run(todo, feed_dict=feed)
        gold = outcomes[0]
        auto = outcomes[1]
        if train:
            loss += outcomes[2]
        for g, a, words in zip(gold, auto, x):
            for gt, at in zip(g, a):
                if gt != 0:
                    total += 1
                    if gt == at:
                        match += 1
    return loss, total, match

if __name__ == '__main__':
    main()
