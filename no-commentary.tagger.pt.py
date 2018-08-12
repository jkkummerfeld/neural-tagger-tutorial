# PyTorch Implementation
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

import torch
torch.manual_seed(0)

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
    model = TaggerModel(NWORDS, NTAGS, pretrained_list, id_to_token)
    # Create optimizer and configure the learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY)
    rescale_lr = lambda epoch: 1 / (1 + LEARNING_DECAY_RATE * epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lr_lambda=rescale_lr)

    expressions = (model, optimizer)
    for epoch in range(EPOCHS):
        random.shuffle(train)

        # Update learning rate
        scheduler.step()

        model.train() 
        model.zero_grad()
        loss, tacc = do_pass(train, token_to_id, tag_to_id, expressions,
                True)

        model.eval() 
        _, dacc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
        print("{} loss {} t-acc {} d-acc {}".format(epoch, loss,
            tacc, dacc))

    # Save model
    torch.save(model.state_dict(), "tagger.pt.model")

    # Load model
    model.load_state_dict(torch.load('tagger.pt.model'))

    # Evaluation pass.
    _, test_acc = do_pass(dev, token_to_id, tag_to_id, expressions, False)
    print("Test Accuracy: {:.3f}".format(test_acc))

class TaggerModel(torch.nn.Module):
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
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
                dropped_word_vectors, lengths, True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,
                batch_first=True, total_length=max_length)
        # Apply dropout
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(lstm_out_dropped)

        # Calculate loss and predictions
        output_scores = output_scores.view(cur_batch_size * max_length, -1)
        flat_labels = labels.view(cur_batch_size * max_length)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss = loss_function(output_scores, flat_labels)
        predicted_tags  = torch.argmax(output_scores, 1)
        predicted_tags = predicted_tags.view(cur_batch_size, max_length)
        return loss, predicted_tags

def do_pass(data, token_to_id, tag_to_id, expressions, train):
    model, optimizer = expressions

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

        # Prepare inputs
        cur_batch_size = len(batch)
        max_length = len(batch[0][0])
        lengths = [len(v[0]) for v in batch]
        input_array = torch.zeros((cur_batch_size, max_length)).long()
        output_array = torch.zeros((cur_batch_size, max_length)).long()
        for n, (tokens, tags) in enumerate(batch):
            token_ids = [token_to_id.get(simplify_token(t), 0) for t in tokens]
            tag_ids = [tag_to_id[t] for t in tags]

            input_array[n, :len(tokens)] = torch.LongTensor(token_ids)
            output_array[n, :len(tags)] = torch.LongTensor(tag_ids)

        # Construct computation
        batch_loss, output = model(input_array, output_array, lengths,
                cur_batch_size)

        # Run computations
        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
            loss += batch_loss.item()
        predicted = output.cpu().data.numpy()

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
