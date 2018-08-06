#### <h1>Implementing a Part-of-Speech tagger in DyNet</h1>
#### <div class=header>
#### This is a top level description
#### </div>
#!/usr/bin/env python3

#### We import libraries for a few specific uses:
#### - argparse for processing command line arguments
#### - random for shuffling our data
#### - sys for flushing output
#### - numpy for handling vectors of data
import argparse
import random
import sys

import numpy as np

#### These are all the constants used in this program
#### Typically, we would make many of these command line arguments and tune using the development set. For simplicity, I have fixed their values here as defined by the Jiang, Liang and Zhang (CoLing 2018). The meaning of each constant is discussed when it is used below.
PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 100
LSTM_HIDDEN = 100 # based on NCRFpp (200 in the paper, but 100 per direction in code)
BATCH_SIZE = 10
LEARNING_RATE = 0.015
LEARNING_DECAY_RATE = 0.05
EPOCHS = 100
KEEP_PROB = 0.5
GLOVE = "../data/glove.6B.100d.txt"
WEIGHT_DECAY = 1e-8

#### DyNet specfic imports
#### The first allows us to configure DyNet from within code rather than on the command line:
#### mem - The amount of system memory initially allocated (DyNet has its own memory management).
#### autobatch - DyNet can automatically batch computations by setting this flag.
#### weight_decay - After every update, multiply the parameter by (1-decay).
#### random_seed - Set the seed for random number generation.
import dynet_config
dynet_config.set(mem=256, autobatch=0, weight_decay=WEIGHT_DECAY,random_seed=0)
# dynet_config.set_gpu() 
import dynet as dy 

#### Reading the data
#### We are expecting a minor variation on the raw Penn Treebank data, with one line per sentence, tokens separated by spaces, and the tag for each token placed next to its word (the | works as a separator as it does not appear as a token).
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
    #### For the purpose of this example we only have arguments for locations of the data.
    parser = argparse.ArgumentParser(description='Dynet tagger.')
    parser.add_argument('training_data')
    parser.add_argument('dev_data')
    args = parser.parse_args()

    #### Read data (see function above)
    train = read_data(args.training_data)
    dev = read_data(args.dev_data)

    #### Make indices
    #### These are mappings from strings to integers that will be used to get the input for our model and to process the output.
    #### UNK is added to our mapping so that there is a vector we can use when we encounter unknown words.
    #### The special PAD symbol is used in PyTorch and Tensorflow as part of shaping the data in a batch to be a consistent size. It is not needed for DyNet, but kepy for consistency.
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
    for tokens, tags in train + dev: # dev is necessary here to get the GloVe embeddings for words in dev but not train loaded. They will not be updated during training.
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

    #### Load pre-trained vectors
    #### I am assuming these are the 50-dimensional GloVe embeddings in their standard format.
    pretrained = {}
    for line in open(GLOVE):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
    #### We will need the word vectors as a list to initialise the embeddings, where each entry in the list corresponds to the token with that index.
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING) # From Jiang, Liang and Zhang
    for word in id_to_token:
        if word.lower() in pretrained: # applying lower() here because all GloVe vectors are for lowercase words
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            #### For words that do not appear in GloVe we generate a random vector (note, the choice of scale here is important).
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)

    #### DyNet model creation
    model = dy.ParameterCollection()
    trainer = dy.SimpleSGDTrainer(model, learning_rate=LEARNING_RATE)
    trainer.set_clip_threshold(-1) # DyNet clips gradients by default, this deactivates that behaviour
    pEmbedding = model.add_lookup_parameters((NWORDS, DIM_EMBEDDING))
    pEmbedding.init_from_array(np.array(pretrained_list))

    stdv = 1.0 / np.sqrt(LSTM_HIDDEN) # Needed to match PyTorch
    f_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_HIDDEN, model,
            forget_bias=(np.random.random_sample() - 0.5) * 2 *stdv)
    b_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_HIDDEN, model,
            forget_bias=(np.random.random_sample() - 0.5) * 2 *stdv)
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
###    # Setting recurrent dropout
###        f_lstm.set_dropouts(1.0 - KEEP_PROB, 1.0 - KEEP_PROB)
###        b_lstm.set_dropouts(1.0 - KEEP_PROB, 1.0 - KEEP_PROB)
    f_lstm.set_dropouts(0.0, 0.0)
    b_lstm.set_dropouts(0.0, 0.0)
    pOutput = model.add_parameters((NTAGS, 2 * LSTM_HIDDEN))
    expressions = (pEmbedding, pOutput, f_lstm, b_lstm, trainer)

    #### Main training loop
    for epoch_no in range(EPOCHS):
        random.shuffle(train)
        trainer.learning_rate = \
                LEARNING_RATE / (1 + LEARNING_DECAY_RATE * epoch_no)
        train_loss, train_total, train_match = do_pass(train, token_to_id,
                tag_to_id, id_to_tag, expressions, True)
        _, dev_total, dev_match = do_pass(dev, token_to_id, tag_to_id,
                id_to_tag, expressions)
        print("epoch {} t-loss {} t-acc {} d-acc {}".format(epoch_no,
            train_loss, train_match / train_total, dev_match / dev_total))

    # TODO: Save model

    # TODO: Reload model

    #### Do evaluation
    _, test_total, test_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag,
            expressions)
    print("Test Accuracy: {:.3f}".format(test_match / test_total))

def do_pass(data, token_to_id, tag_to_id, id_to_tag, expressions, train=False):
    pEmbedding, pOutput, f_lstm, b_lstm, trainer = expressions

    #### Loop over batches
    loss = 0
    match = 0
    total = 0
    start = 0
    while start < len(data):
        #### Form batch
        batch = data[start : start + BATCH_SIZE]
        batch.sort(key = lambda x: -len(x[0]))
        start += BATCH_SIZE
        if start % 4000 == 0:
            print(loss, match / total)
            sys.stdout.flush()

        #### DyNet network construction
        dy.renew_cg()
        errs = []
        predicted = []
        for n, (tokens, tags) in enumerate(batch):
            # Convert to indices
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            # Decode and update
            f_init = f_lstm.initial_state()
            b_init = b_lstm.initial_state()
            wembs = [dy.lookup(pEmbedding, w) for w in token_ids]
            if train:
                wembs = [dy.dropout(w, 1.0 - KEEP_PROB) for w in wembs]
            f_lstm_output = [x.output() for x in f_init.add_inputs(wembs)]
            rev_embs = reversed(wembs)
            b_lstm_output = [x.output() for x in b_init.add_inputs(rev_embs)]

            pred_tags = []
            for f, b, t in zip(f_lstm_output, reversed(b_lstm_output), tag_ids):
                combined = dy.concatenate([f,b])
                if train:
                    combined = dy.dropout(combined, 1.0 - KEEP_PROB)
                r_t = pOutput * combined
                if train:
                    err = dy.pickneglogsoftmax(r_t, t)
                    errs.append(err)
                out = dy.softmax(r_t)
                chosen = np.argmax(out.npvalue())
                pred_tags.append(chosen)
            predicted.append(pred_tags)

        #### During training, do update
        if train:
            sum_errs = dy.esum(errs)
            loss += sum_errs.scalar_value()
            sum_errs.backward()
            trainer.update()

        #### Scoring
        for (_, g), a in zip(batch, predicted):
            total += len(g)
            for gt, at in zip(g, a):
                gt = tag_to_id[gt]
                if gt == at:
                    match += 1

    return loss, total, match

if __name__ == '__main__':
    main()
