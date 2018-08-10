#### Imports
#### We use argparse for processing command line arguments, random for shuffling our data, sys for flushing output, and numpy for handling vectors of data.
# PyTorch Implementation
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

#### PyTorch specfic imports
import torch
torch.manual_seed(0)

#### Data reading
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

#### Simplificaiton by replacing all digits with 0 to decrease sparsity.
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
    # PyTorch model creation
    model = TaggerModel(NWORDS, NTAGS, pretrained_list, id_to_token)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)
    #### The learning rate for each epoch is set by multiplying the inital rate by the factor produced by this function.
    rescale_lr = lambda epoch: 1 / (1 + LEARNING_DECAY_RATE * epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=rescale_lr)

    #### Main training loop, in which we shuffle the data, set the learning rate, do one complete pass over the training data, then evaluate on the development data.
    #### To make the code match across the three versions, we group together some framework specifc values needed when doing a pass over the data.
    expressions = (model, optimizer)
    for epoch in range(EPOCHS):
        random.shuffle(train)

        #### Determine the current learning rate
        scheduler.step() # First call to rescale_lr is with a 0, so this should be done here

        #### Set in training mode, which does things like enable dropout components, and initialise the gradient to zero.
        model.train() 
        model.zero_grad()
        #### Training pass
        loss, tacc = do_pass(train, token_to_id, tag_to_id, expressions,
                True)

        #### Set in evaluation mode, which does things like disable dropout components
        model.eval() 
        #### Dev pass
        _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
        print("{} loss {} t-acc {} d-acc {}".format(epoch, loss,
            tacc, dacc))

    #### Save and load model. Both must be done after the definitions above (ie, the model should be recreated, then have its parameters set to match this saved version).
    torch.save(model.state_dict(), "tagger.pt.model")
    model.load_state_dict(torch.load('tagger.pt.model'))

    #### Evaluation
    _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
    print("Test Accuracy: {:.3f}".format(test_acc))

#### Neural network definition code. In PyTorch networks are defined using classes that extend Module.
class TaggerModel(torch.nn.Module):
    #### In the constructor we define objects that will do each of the computations
    def __init__(self, nwords, ntags, pretrained_list, id_to_token):
        super().__init__()

        #### Convert the word embeddings into a PyTorch tensor
        pretrained_tensor = torch.FloatTensor(pretrained_list) # TODO: , sparse=True) Doesn't work?
        self.word_embedding = torch.nn.Embedding.from_pretrained(
                pretrained_tensor, freeze=False)
        self.word_dropout = torch.nn.Dropout(1 - KEEP_PROB)

        self.lstm = torch.nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, num_layers=1,
                batch_first=True, bidirectional=True)
        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_PROB)

        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN * 2, ntags)

    def forward(self, sentences, labels, lengths, cur_batch_size):
        max_length = sentences.size(1)

        #### Look up word vectors
        word_vectors = self.word_embedding(sentences)
        #### Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)

        #### Assuming the data is ordered longest to shortest, this provides a view of the data that fits with how cuDNN works
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
                dropped_word_vectors, lengths, True)
        #### Run the LSTM over the input
        lstm_out, _ = self.lstm(packed_words, None)
        #### Reverse the view shift made for cuDNN. Specifying total_length is not necessary in general (it can be inferred), but is necessary for parallel processing.
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                batch_first=True, total_length=max_length)
        #### Apply dropout to the output
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        #### Matrix multiply to get distribution over tags
        output_scores = self.hidden_to_tag(lstm_out_dropped)

        #### Reshape to [batch size * sequence length, ntags] for more efficient processing
        output_scores = output_scores.view(cur_batch_size * max_length, -1)
        flat_labels = labels.view(cur_batch_size * max_length)
        #### Calculate the cross entropy loss, ignoring padding, and summing losses across the batch
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss = loss_function(output_scores, flat_labels)
        #### Identify the highest scoring tag in each case and reshape to be [batch, sequence]
        _, predicted_tags  = torch.max(output_scores, 1)
        predicted_tags = predicted_tags.view(cur_batch_size, max_length)
        return loss, predicted_tags

#### Inference (the same function for train and test)
def do_pass(data, token_to_id, tag_to_id, expressions, train):
    model, optimizer = expressions

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

        #### Prepare input arrays, using .long() to cast the type from Tensor to LongTensor.
        cur_batch_size = len(batch)
        max_length = len(batch[0][0])
        lengths = [len(v[0]) for v in batch]
        input_array = torch.zeros((cur_batch_size, max_length)).long()
        output_array = torch.zeros((cur_batch_size, max_length)).long()
        for n, (tokens, tags) in enumerate(batch):
            #### Using the indices we map our srings to numbers.
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]
            #### Fill the arrays, leaving the remaining values as zero (our padding value).
            input_array[n, :len(tokens)] = torch.LongTensor(token_ids)
            output_array[n, :len(tags)] = torch.LongTensor(tag_ids)

        #### Calling the model as a function will run its forward() function, which constructs the network.
        batch_loss, output = model(input_array, output_array, lengths,
                cur_batch_size)

        #### In training we do the backwards pass, apply the update, and reset the gradient.
        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
            #### To get the loss value we use .item().
            loss += batch_loss.item()
        #### Our output is an array (rather than a single value), so we use a different approach to get it into a usable form.
        predicted = output.cpu().data.numpy()

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
