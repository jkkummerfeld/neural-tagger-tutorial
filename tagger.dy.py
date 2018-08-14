#### We use argparse for processing command line arguments, random for shuffling our data, sys for flushing output, and numpy for handling vectors of data.
# DyNet Implementation
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
WEIGHT_DECAY = 1e-8 # WEIGHT_DECAY - part of a rescaling of weights when an update occurs.

#### Dynet library imports. The first allows us to configure DyNet from within code rather than on the command line: mem is the amount of system memory initially allocated (DyNet has its own memory management), autobatch toggles automatic parallelisation of computations, weight_decay rescales weights by (1 - decay) after every update, random_seed sets the seed for random number generation.
import dynet_config
dynet_config.set(mem=256, autobatch=0, weight_decay=WEIGHT_DECAY,random_seed=0)
# dynet_config.set_gpu() for when we want to run with GPUs
import dynet as dy 

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
    model = dy.ParameterCollection()
    # Create word embeddings and initialise
    #### Lookup parameters are a matrix that supports efficient sparse lookup.
    pEmbedding = model.add_lookup_parameters((NWORDS, DIM_EMBEDDING))
    pEmbedding.init_from_array(np.array(pretrained_list))
    # Create LSTM parameters
    #### Objects that create LSTM cells and the necessary parameters.
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
    # Create the trainer
    trainer = dy.SimpleSGDTrainer(model, learning_rate=LEARNING_RATE)
    #### DyNet clips gradients by default, which we disable here (this can have a big impact on performance).
    trainer.set_clip_threshold(-1)

    #### To make the code match across the three versions, we group together some framework specific values needed when doing a pass over the data.
    expressions = (pEmbedding, pOutput, f_lstm, b_lstm, trainer)
    #### Main training loop, in which we shuffle the data, set the learning rate, do one complete pass over the training data, then evaluate on the development data.
    for epoch in range(EPOCHS):
        random.shuffle(train)

        ####
        # Update learning rate
        trainer.learning_rate = LEARNING_RATE / (1+ LEARNING_DECAY_RATE * epoch)

        #### Training pass.
        loss, tacc = do_pass(train, token_to_id, tag_to_id, expressions, True)
        #### Dev pass.
        _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
        print("{} loss {} t-acc {} d-acc {}".format(epoch, loss, tacc, dacc))

    #### The syntax varies, but in all three cases either saving or loading the parameters of a model must be done after the model is defined.
    # Save model
    model.save("tagger.dy.model")

    # Load model
    model.populate("tagger.dy.model")

    # Evaluation pass.
    _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
    print("Test Accuracy: {:.3f}".format(test_acc))

#### Inference (the same function for train and test).
def do_pass(data, token_to_id, tag_to_id, expressions, train):
    pEmbedding, pOutput, f_lstm, b_lstm, trainer = expressions

    # Loop over batches
    loss = 0
    match = 0
    total = 0
    start = 0
    while start < len(data):
        #### Form the batch and order it based on length (important for efficient processing in PyTorch).
        batch = data[start : start + BATCH_SIZE]
        batch.sort(key = lambda x: -len(x[0]))
        start += BATCH_SIZE
        #### Log partial results so we can conveniently check progress.
        if start % 4000 == 0:
            print(loss, match / total)
            sys.stdout.flush()

        #### Start a new computation graph for this batch.
        # Process batch
        dy.renew_cg()
        #### For each example, we will construct an expression that gives the loss.
        loss_expressions = []
        predicted = []
        #### Convert tokens and tags from strings to numbers using the indices.
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            #### Now we define the computation to be performed with the model. Note that they are not applied yet, we are simply building the computation graph.
            # Look up word embeddings
            wembs = [dy.lookup(pEmbedding, w) for w in token_ids]
            # Apply dropout
            if train:
                wembs = [dy.dropout(w, 1.0 - KEEP_PROB) for w in wembs]
            # Feed words into the LSTM
            #### Create an expression for two LSTMs and feed in the embeddings (reversed in one case).
            #### We pull out the output vector from the cell state at each step.
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
                    #### We are not actually evaluating the loss values here, instead we collect them together in a list. This enables DyNet's <a href="http://dynet.readthedocs.io/en/latest/tutorials_notebooks/Autobatching.html">autobatching</a>.
                    loss_expressions.append(err)
                # Calculate the highest scoring tag
                #### This call to .npvalue() will lead to evaluation of the graph and so we don't actually get the benefits of autobatching. With some refactoring we could get the benefit back (simply keep the r_t expressions around and do this after the update), but that would have complicated this code.
                chosen = np.argmax(r_t.npvalue())
                pred_tags.append(chosen)
            predicted.append(pred_tags)

        # combine the losses for the batch, do an update, and record the loss
        if train:
            loss_for_batch = dy.esum(loss_expressions)
            loss_for_batch.backward()
            trainer.update()
            loss += loss_for_batch.scalar_value()

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
