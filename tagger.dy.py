#!/usr/bin/env python3

import argparse
import random
import sys

import numpy as np

import dynet_config
dynet_config.set(mem=2048, autobatch=1, weight_decay=1e-8)

import dynet as dy 


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
    parser = argparse.ArgumentParser(description='Dynet tagger.')
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

    # Load pre-trained vectors
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        if word in token_to_id:
            vector = np.array([float(v) for v in parts[1:]])
            pretrained[word] = vector
            if word not in token_to_id:
                token_to_id[token] = len(id_to_token)
                id_to_token.append(token)

    NWORDS = len(id_to_token)
    NTAGS = len(id_to_tag)

    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model, learning_rate=LEARNING_RATE)
    pEmbedding = model.add_lookup_parameters((NWORDS, DIM_EMBEDDING))
    f_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_SIZES[0], model)
    b_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_SIZES[0], model)
    pOutput = model.add_parameters((NTAGS, 2 * LSTM_SIZES[0]))
    expressions = (pEmbedding, pOutput, f_lstm, b_lstm, trainer)

    pretrained_array = []
    for word in id_to_token:
        if word in pretrained:
            pretrained_array.append(pretrained[word])
        elif word.lower() in pretrained:
            pretrained_array.append(pretrained[word.lower()])
        else:
            pretrained_array.append(pEmbedding.row_as_array(token_to_id[word]))
    pEmbedding.init_from_array(np.array(pretrained_array))

    for epoch_no in range(EPOCHS):
        random.shuffle(train)
        trainer.learning_rate = LEARNING_RATE / (1 + LEARNING_DECAY_RATE * epoch_no)

        # do iteration
        train_loss, train_total, train_match = do_pass(train, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, True)
        _, dev_total, dev_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions)
        print("epoch {} t-loss {} t-acc {} d-acc {}".format(epoch_no, train_loss, train_match / train_total, dev_match / dev_total))

    # Save model

    # Reload model

    # do evaluation
    _, test_total, test_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions)
    print("Test Accuracy: {:.3f}".format(test_match / test_total))

def do_pass(data, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, train=False):
    pEmbedding, pOutput, f_lstm, b_lstm, trainer = expressions

    loss = 0
    match = 0
    total = 0
    start = 0
    while start + BATCH_SIZE < len(data):
        batch = data[start : start + BATCH_SIZE]
        start += BATCH_SIZE
        if start % 4000 == 0:
            print(start, loss, match / total)
            sys.stdout.flush()

        dy.renew_cg()
        errs = []
        predicted = []
        for tokens, tags in batch:
            # Convert to indices
            token_ids = [token_to_id.get(t, token_to_id[UNK]) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            # Decode and update
            if train and KEEP_PROB < 1.0:
                f_lstm.set_dropouts(1.0 - KEEP_PROB, 1.0 - KEEP_PROB)
                b_lstm.set_dropouts(1.0 - KEEP_PROB, 1.0 - KEEP_PROB)

            f_init = f_lstm.initial_state()
            b_init = b_lstm.initial_state()
            wembs = [dy.lookup(pEmbedding, w) for w in token_ids]
            f_lstm_output = [x.output() for x in f_init.add_inputs(wembs)]
            b_lstm_output = [x.output() for x in b_init.add_inputs(reversed(wembs))]

            O = dy.parameter(pOutput)

            pred_tags = []
            for f, b, t in zip(f_lstm_output, b_lstm_output, tag_ids):
                combined = dy.concatenate([f,b])
                r_t = O * combined
                if train:
                    err = dy.pickneglogsoftmax(r_t, t)
                    errs.append(err)
                out = dy.softmax(r_t)
                chosen = np.argmax(out.npvalue())
                pred_tags.append(chosen)
            predicted.append(pred_tags)

        if train:
            sum_errs = dy.esum(errs)
            loss += sum_errs.scalar_value()
            sum_errs.backward()
            trainer.update()

        for (_, g), a in zip(batch, predicted):
            total += len(g)
            for gt, at in zip(g, a):
                gt = tag_to_id[gt]
                if gt == at:
                    match += 1

    return loss, total, match

if __name__ == '__main__':
    main()
