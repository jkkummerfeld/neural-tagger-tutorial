#### We use argparse for processing command line arguments, random for shuffling our data, sys for flushing output, and numpy for handling vectors of data.
# PyTorch Implementation
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

#### PyTorch library import.
import torch
torch.manual_seed(0)

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
    model = TaggerModel(NWORDS, NTAGS, pretrained_list, id_to_token)
    # Create optimizer and configure the learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)
    #### The learning rate for each epoch is set by multiplying the initial rate by the factor produced by this function.
    rescale_lr = lambda epoch: 1 / (1 + LEARNING_DECAY_RATE * epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=rescale_lr)

    #### To make the code match across the three versions, we group together some framework specific values needed when doing a pass over the data.
    expressions = (model, optimizer)
    #### Main training loop, in which we shuffle the data, set the learning rate, do one complete pass over the training data, then evaluate on the development data.
    for epoch in range(EPOCHS):
        random.shuffle(train)

        ####
        # Update learning rate
        #### First call to rescale_lr is with a 0, which is why this must be done before the pass over the data.
        scheduler.step()

        #### Training mode (and evaluation mode below) do things like enable dropout components.
        model.train() 
        model.zero_grad()
        #### Training pass.
        loss, tacc = do_pass(train, token_to_id, tag_to_id, expressions,
                True)

        ####
        model.eval() 
        #### Dev pass.
        _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
        print("{} loss {} t-acc {} d-acc {}".format(epoch, loss,
            tacc, dacc))

    #### The syntax varies, but in all three cases either saving or loading the parameters of a model must be done after the model is defined.
    # Save model
    torch.save(model.state_dict(), "tagger.pt.model")

    # Load model
    model.load_state_dict(torch.load('tagger.pt.model'))

    # Evaluation pass.
    _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
    print("Test Accuracy: {:.3f}".format(test_acc))

#### Neural network definition code. In PyTorch networks are defined using classes that extend Module.
class TaggerModel(torch.nn.Module):
    #### In the constructor we define objects that will do each of the computations.
    def __init__(self, nwords, ntags, pretrained_list, id_to_token):
        super().__init__()

        # Create word embeddings
        pretrained_tensor = torch.FloatTensor(pretrained_list)
        self.word_embedding = torch.nn.Embedding.from_pretrained(
                pretrained_tensor, freeze=False)
        # Create input dropout parameter
        self.word_dropout = torch.nn.Dropout(1 - KEEP_PROB)
        # Create LSTM parameters
        self.lstm = torch.nn.LSTM(DIM_EMBEDDING, LSTM_HIDDEN, num_layers=1,
                batch_first=True, bidirectional=True)
        # Create output dropout parameter
        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_PROB)
        # Create final matrix multiply parameters
        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN * 2, ntags)

    def forward(self, sentences, labels, lengths, cur_batch_size):
        max_length = sentences.size(1)

        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)
        # Run the LSTM over the input, reshaping data for efficiency
        #### Assuming the data is ordered longest to shortest, this provides a view of the data that fits with how cuDNN works.
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
                dropped_word_vectors, lengths, True)
        #### The None argument is an optional initial hidden state (default is a zero vector). The ignored return value contains the hidden states.
        lstm_out, _ = self.lstm(packed_words, None)
        #### Reverse the view shift made for cuDNN. Specifying total_length is not necessary in general (it can be inferred), but is necessary for parallel processing. The ignored return value contains the length of each sequence.
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                batch_first=True, total_length=max_length)
        # Apply dropout
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(lstm_out_dropped)

        # Calculate loss and predictions
        #### We reshape to [batch size * sequence length , ntags] for more efficient processing.
        output_scores = output_scores.view(cur_batch_size * max_length, -1)
        flat_labels = labels.view(cur_batch_size * max_length)
        #### The ignore index refers to outputs to not score, which we use to ignore padding. 'reduction' defines how to combine the losses at each point in the sequence. The default is elementwise_mean, which would not do what we want.
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss = loss_function(output_scores, flat_labels)
        predicted_tags  = torch.argmax(output_scores, 1)
        #### Reshape to have dimensions [batch size , sequence length].
        predicted_tags = predicted_tags.view(cur_batch_size, max_length)
        return loss, predicted_tags

#### Inference (the same function for train and test).
def do_pass(data, token_to_id, tag_to_id, expressions, train):
    model, optimizer = expressions

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

        ####
        # Prepare inputs
        #### Prepare input arrays, using .long() to cast the type from Tensor to LongTensor.
        cur_batch_size = len(batch)
        max_length = len(batch[0][0])
        lengths = [len(v[0]) for v in batch]
        input_array = torch.zeros((cur_batch_size, max_length)).long()
        output_array = torch.zeros((cur_batch_size, max_length)).long()
        #### Convert tokens and tags from strings to numbers using the indices.
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            #### Fill the arrays, leaving the remaining values as zero (our padding value).
            input_array[n, :len(tokens)] = torch.LongTensor(token_ids)
            output_array[n, :len(tags)] = torch.LongTensor(tag_ids)

        # Construct computation
        #### Calling the model as a function will run its forward() function, which constructs the computations.
        batch_loss, output = model(input_array, output_array, lengths,
                cur_batch_size)

        # Run computations
        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
            #### To get the loss value we use .item().
            loss += batch_loss.item()
        #### Our output is an array (rather than a single value), so we use a different approach to get it into a usable form.
        predicted = output.cpu().data.numpy()

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
