#!/usr/bin/env python3

import argparse
import random

import numpy as np
import dynet as dy

PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 128
LSTM_SIZES = [50]
BATCH_SIZE = 1
LEARNING_RATE = 0.1
EPOCHS = 50
KEEP_PROB = 0.5

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

    NWORDS = len(id_to_token)
    NTAGS = len(id_to_tag)

    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model, learning_rate=0.1)
    pEmbedding = model.add_lookup_parameters((NWORDS, DIM_EMBEDDING))
    lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_SIZES[0], model)
    pOutput = model.add_parameters((NTAGS, LSTM_SIZES[0]))
    expressions = (pEmbedding, pOutput, lstm, trainer)

    for epoch_no in range(EPOCHS):
        random.shuffle(train)

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
    pEmbedding, pOutput, lstm, trainer = expressions

    loss = 0
    match = 0
    total = 0
    start = 0
    while start + BATCH_SIZE < len(data):
        batch = [v for v in data[start : start + BATCH_SIZE]]
        start += BATCH_SIZE

        errs = []
        predicted = []
        for tokens, tags in batch:
            # Convert to indices
            token_ids = [token_to_id.get(t, token_to_id[UNK]) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            # Decode and update
            dy.renew_cg()

            if train and KEEP_PROB < 1.0:
                lstm.set_dropouts(1.0 - KEEP_PROB, 1.0 - KEEP_PROB)

            f_init = lstm.initial_state()
            wembs = [dy.lookup(pEmbedding, w) for w in token_ids]
            lstm_output = [x.output() for x in f_init.add_inputs(wembs)]
            O = dy.parameter(pOutput)

            pred_tags = []
            for f, t in zip(lstm_output, tag_ids):
                r_t = O * f
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
