# DyNet Implementation
import argparse
import random
import sys

import numpy as np

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
WEIGHT_DECAY = 1e-8 # WEIGHT_DECAY - part of a rescaling of weights when an update occurs.

import dynet_config
dynet_config.set(mem=256, autobatch=0, weight_decay=WEIGHT_DECAY,random_seed=0)
# dynet_config.set_gpu() 
import dynet as dy 

# Data reading
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

def simplify_token(token):
    chars = []
    for char in token:
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)

def main():
    parser = argparse.ArgumentParser(description='POS tagger.')
    parser.add_argument('training_data')
    parser.add_argument('dev_data')
    args = parser.parse_args()

    train = read_data(args.training_data)
    dev = read_data(args.dev_data)

    # Make indices
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
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
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING)
    for word in id_to_token:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)

    # Model creation
    model = dy.ParameterCollection()
    # Create word embeddings and initialise
    pEmbedding = model.add_lookup_parameters((NWORDS, DIM_EMBEDDING))
    pEmbedding.init_from_array(np.array(pretrained_list))
    # Create LSTM parameters
    stdv = 1.0 / np.sqrt(LSTM_HIDDEN) # Needed to match PyTorch
    f_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_HIDDEN, model,
            forget_bias=(np.random.random_sample() - 0.5) * 2 * stdv)
    b_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_HIDDEN, model,
            forget_bias=(np.random.random_sample() - 0.5) * 2 * stdv)
    # Create output layer
    pOutput = model.add_parameters((NTAGS, 2 * LSTM_HIDDEN))
    
    # Set recurrent dropout values (not used in this case)
    f_lstm.set_dropouts(0.0, 0.0)
    b_lstm.set_dropouts(0.0, 0.0)
    # Initialise LSTM parameters
    f_lstm.get_parameters()[0][0].set_value(
            np.random.uniform(-stdv, stdv, [4 * LSTM_HIDDEN, DIM_EMBEDDING]))
    f_lstm.get_parameters()[0][1].set_value(
            np.random.uniform(-stdv, stdv, [4 * LSTM_HIDDEN, LSTM_HIDDEN]))
    f_lstm.get_parameters()[0][2].set_value(
            np.random.uniform(-stdv, stdv, [4 * LSTM_HIDDEN]))
    b_lstm.get_parameters()[0][0].set_value(
            np.random.uniform(-stdv, stdv, [4 * LSTM_HIDDEN, DIM_EMBEDDING]))
    b_lstm.get_parameters()[0][1].set_value(
            np.random.uniform(-stdv, stdv, [4 * LSTM_HIDDEN, LSTM_HIDDEN]))
    b_lstm.get_parameters()[0][2].set_value(
            np.random.uniform(-stdv, stdv, [4 * LSTM_HIDDEN]))

    # Create the trainer
    trainer = dy.SimpleSGDTrainer(model, learning_rate=LEARNING_RATE)
    trainer.set_clip_threshold(-1)

    expressions = (pEmbedding, pOutput, f_lstm, b_lstm, trainer)
    for epoch in range(EPOCHS):
        random.shuffle(train)

        # Update learning rate
        trainer.learning_rate = LEARNING_RATE / (1+ LEARNING_DECAY_RATE * epoch)

        loss, tacc = do_pass(train, token_to_id, tag_to_id, expressions, True,
                current_lr)
        _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
        print("{} loss {} t-acc {} d-acc {}".format(epoch, loss, tacc, dacc))

    # Save model
    model.save("tagger.dy.model")

    # Load model
    model.populate("tagger.dy.model")

    # Evaluation pass.
    _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
    print("Test Accuracy: {:.3f}".format(test_acc))

def do_pass(data, token_to_id, tag_to_id, expressions, train):
    pEmbedding, pOutput, f_lstm, b_lstm, trainer = expressions

    # Loop over batches
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
            sys.stdout.flush()

        # Process batch
        dy.renew_cg()
        loss_expressions = []
        predicted = []
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            # Look up word embeddings
            wembs = [dy.lookup(pEmbedding, w) for w in token_ids]
            # Apply dropout
            if train:
                wembs = [dy.dropout(w, 1.0 - KEEP_PROB) for w in wembs]
            # Feed words into the LSTM
            f_init = f_lstm.initial_state()
            f_lstm_output = [x.output() for x in f_init.add_inputs(wembs)]
            rev_embs = reversed(wembs)
            b_init = b_lstm.initial_state()
            b_lstm_output = [x.output() for x in b_init.add_inputs(rev_embs)]

            # For each output, calculate the output and loss
            pred_tags = []
            for f, b, t in zip(f_lstm_output, reversed(b_lstm_output), tag_ids):
                # Combine the outputs
                combined = dy.concatenate([f,b])
                # Apply dropout
                if train:
                    combined = dy.dropout(combined, 1.0 - KEEP_PROB)
                # Matrix multiply to get scores for each tag
                r_t = pOutput * combined
                # Calculate cross-entropy loss
                if train:
                    err = dy.pickneglogsoftmax(r_t, t)
                    loss_expressions.append(err)
                # Calculate the highest scoring tag
                chosen = np.argmax(r_t.npvalue())
                pred_tags.append(chosen)
            predicted.append(pred_tags)

        # combine the losses for the batch, do an update, and record the loss
        if train:
            loss_for_batch = dy.esum(loss_expressions)
            loss_for_batch.backward()
            trainer.update()
            loss += loss_for_batch.scalar_value()

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
