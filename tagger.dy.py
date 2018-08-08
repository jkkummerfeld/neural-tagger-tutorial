#### Imports
#### We use argparse for processing command line arguments, random for shuffling our data, sys for flushing output, and numpy for handling vectors of data.
import argparse
import random
import sys

import numpy as np

#### Constants
#### Typically, we would make many of these command line arguments and tune using the development set. For simplicity, I have fixed their values here to match Jiang, Liang and Zhang (CoLing 2018).
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

#### DyNet specfic imports
#### The first allows us to configure DyNet from within code rather than on the command line:  mem is the amount of system memory initially allocated (DyNet has its own memory management), autobatch toggles automatic parallelisation of computations, weight_decay rescales weights by (1 - decay) after every update, random_seed sets the seed for random number generation.
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
    parser = argparse.ArgumentParser(description='POS tagger.')
    parser.add_argument('training_data')
    parser.add_argument('dev_data')
    args = parser.parse_args()

    #### Read data (see function above)
    train = read_data(args.training_data)
    dev = read_data(args.dev_data)

    #### Make indices
    #### These are mappings from strings to integers that will be used to get the input for our model and to process the output. UNK is added to our mapping so that there is a vector we can use when we encounter unknown words. The special PAD symbol is used in PyTorch and Tensorflow as part of shaping the data in a batch to be a consistent size. It is not needed for DyNet, but kept for consistency.
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
    #### dev is necessary here to get the GloVe embeddings for words in dev but not train loaded. They will not be updated during training as they do not occur.
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

    #### Load pre-trained vectors
    #### I am assuming these are the 100-dimensional GloVe embeddings in their standard format.
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

    ####
    # DyNet model creation
    model = dy.ParameterCollection()
    #### Lookup parameters are a matrix that supports efficient sparse lookup.
    pEmbedding = model.add_lookup_parameters((NWORDS, DIM_EMBEDDING))
    pEmbedding.init_from_array(np.array(pretrained_list))
    #### Objects that create LSTM cells and the necessary parameters.
    stdv = 1.0 / np.sqrt(LSTM_HIDDEN) # Needed to match PyTorch
    f_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_HIDDEN, model,
            forget_bias=(np.random.random_sample() - 0.5) * 2 * stdv)
    b_lstm = dy.VanillaLSTMBuilder(1, DIM_EMBEDDING, LSTM_HIDDEN, model,
            forget_bias=(np.random.random_sample() - 0.5) * 2 * stdv)
    #### A simple weight matrix for the final output calculation.
    pOutput = model.add_parameters((NTAGS, 2 * LSTM_HIDDEN))

    #### Setting recurrent dropout values (not used in this case).
    f_lstm.set_dropouts(0.0, 0.0)
    b_lstm.set_dropouts(0.0, 0.0)
    #### To match PyTorch, we initialise the parameters with an unconventional approach.
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

    #### The trainer object is used to update the model.
    #### DyNet clips gradients by default, which we disable here (this can have a big impact on performance).
    trainer = dy.SimpleSGDTrainer(model, learning_rate=LEARNING_RATE)
    trainer.set_clip_threshold(-1)

    #### Main training loop, in which we shuffle the data, set the learning rate, do one complete pass over the training data, then evaluate on the development data.
    #### To make the code match across the three versions, we group together some framework specifc values needed when doing a pass over the data.
    expressions = (pEmbedding, pOutput, f_lstm, b_lstm, trainer)
    for epoch in range(EPOCHS):
        random.shuffle(train)

        #### Determine the current learning rate
        trainer.learning_rate = LEARNING_RATE / (1+ LEARNING_DECAY_RATE * epoch)

        #### Training pass
        loss, tacc = do_pass(train, token_to_id, tag_to_id, expressions, True,
                current_lr)
        #### Dev pass
        _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
        print("{} loss {} t-acc {} d-acc {}".format(epoch, loss, tacc, dacc))

    #### Save and load model. Both must be done after the definitions above (ie, the model should be recreated, then have its parameters set to match this saved version).
    model.save("tagger.dy.model")
    model.populate("tagger.dy.model")

    #### Evaluation
    _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
    print("Test Accuracy: {:.3f}".format(test_acc))

#### Inference (the same function for train and test)
def do_pass(data, token_to_id, tag_to_id, expressions, train):
    pEmbedding, pOutput, f_lstm, b_lstm, trainer = expressions

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

        #### Start a new computation graph for this batch
        dy.renew_cg()
        #### For each example, we will construct an expression that gives the loss.
        loss_expressions = []
        predicted = []
        for n, (tokens, tags) in enumerate(batch):
            #### Convert tokens and tags from strings to numbers using the indices
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            #### Look up word embeddings
            wembs = [dy.lookup(pEmbedding, w) for w in token_ids]
            #### During training, apply dropout to the inputs
            if train:
                wembs = [dy.dropout(w, 1.0 - KEEP_PROB) for w in wembs]
            #### Create an expression for two LSTMs and feed in the embeddings (reversed in one case).
            #### We pull out the output vector from the cell state at each step.
            f_init = f_lstm.initial_state()
            f_lstm_output = [x.output() for x in f_init.add_inputs(wembs)]
            rev_embs = reversed(wembs)
            b_init = b_lstm.initial_state()
            b_lstm_output = [x.output() for x in b_init.add_inputs(rev_embs)]

            pred_tags = []
            #### Combine the outputs
            for f, b, t in zip(f_lstm_output, reversed(b_lstm_output), tag_ids):
                combined = dy.concatenate([f,b])
                #### Apply dropout to the output
                if train:
                    combined = dy.dropout(combined, 1.0 - KEEP_PROB)
                #### Multiple by a matrix to get scores for each tag
                r_t = pOutput * combined
                if train:
                    #### When training, get an expression for the cross-entropy loss
                    err = dy.pickneglogsoftmax(r_t, t)
                    loss_expressions.append(err)
                #### Calculate the highest scoring tag (which will lead to evaluation of the graph)
                chosen = np.argmax(r_t.npvalue())
                pred_tags.append(chosen)
            predicted.append(pred_tags)

        #### During training, combine the losses for the batch, do an update, and record the loss
        if train:
            loss_for_batch = dy.esum(loss_expressions)
            loss_for_batch.backward()
            trainer.update()
            loss += loss_for_batch.scalar_value()

        #### Update the number of correct tags and total tags
        for (_, g), a in zip(batch, predicted):
            total += len(g)
            for gt, at in zip(g, a):
                gt = tag_to_id[gt]
                if gt == at:
                    match += 1

    return loss, match / total

if __name__ == '__main__':
    main()
