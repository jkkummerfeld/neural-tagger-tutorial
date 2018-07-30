#!/usr/bin/env python3

import argparse
import random
import sys

import numpy as np

import torch

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
WEIGHT_DECAY = 1e-8

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

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
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD]
    tag_to_id = {PAD: 0}
    for tokens, tags in train + dev:
###    for tokens, tags in train:
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
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector

    NWORDS = len(id_to_token)
    NTAGS = len(id_to_tag)

    model = TaggerModel(NWORDS, NTAGS, pretrained, id_to_token)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    expressions = (model,optimizer)

    for epoch_no in range(EPOCHS):
        random.shuffle(train)
        current_lr = LEARNING_RATE / (1 + LEARNING_DECAY_RATE * epoch_no)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # do iteration
        model.train() # Set in training mode, which enables dropout components (for example)
        model.zero_grad() # Set all gradients to 0
        train_loss, train_total, train_match = do_pass(train, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, True)

        model.eval() # Set in evaluation mode, which disables dropout components (for example)
        _, dev_total, dev_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, False)

        print("epoch {} t-loss {} t-acc {} d-acc {}".format(epoch_no, train_loss, train_match / train_total, dev_match / dev_total))

    # Save model

    # Reload model

    # do evaluation
    _, test_total, test_match = do_pass(dev, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions)
    print("Test Accuracy: {:.3f}".format(test_match / test_total))

class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags, pretrained_dict, id_to_token):
        super().__init__()

        pretrained_list = []
        scale = np.sqrt(3.0 / DIM_EMBEDDING) # From Jiang, Liang and Zhang
        for word in id_to_token:
            if word in pretrained_dict:
                pretrained_list.append(pretrained_dict[word])
            elif word.lower() in pretrained_dict:
                pretrained_list.append(pretrained_dict[word.lower()])
            else:
                pretrained_list.append(np.random.uniform(-scale, scale, [DIM_EMBEDDING]))
        pretrained_tensor = torch.FloatTensor(pretrained_list)
        self.word_embedding = torch.nn.Embedding.from_pretrained(pretrained_tensor, freeze=False)# , sparse=True) Doesn't work?

        self.word_dropout = torch.nn.Dropout(1 - KEEP_PROB)

        self.lstm = torch.nn.LSTM(DIM_EMBEDDING, LSTM_SIZES[0], num_layers=1, batch_first=True, bidirectional=True)
        # TODO: How to do recurrent dropout?
        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_PROB)

        self.hidden_to_tag = torch.nn.Linear(LSTM_SIZES[-1] * 2, ntags)

    def forward(self, sentences, labels, lengths, cur_batch_size):
        max_length = sentences.size(1)

        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)

        # Assuming the data is ordered longest to shortest, this provides a view of the data that fits with how cuDNN works
###        packed_words = torch.nn.utils.rnn.pack_sequence(dropped_word_vectors)
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(dropped_word_vectors, lengths, True)
        # Run the LSTM over the input
        lstm_out, _ = self.lstm(packed_words, None)
        # Reverse the view shift made for cuDNN
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=max_length) # Specifying total_length is not necessary in general (it can be inferred), but is important for parallel processing
        # Apply dropout to the output
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        # Matrix multiply to get distribution over tags
        output_scores = self.hidden_to_tag(lstm_out_dropped)

        # Reshape to [batch size * sequence length, ntags] for more efficient processing
        output_scores = output_scores.view(cur_batch_size * max_length, -1)
        flat_labels = labels.view(cur_batch_size * max_length)
        # Calculate the cross entropy loss, ignoring padding, and summing losses across the batch
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss = loss_function(output_scores, flat_labels)
        # Identify the highest scoring tag in each case and reshape to be [batch, sequence]
        _, predicted_tags  = torch.max(output_scores, 1)
        predicted_tags = predicted_tags.view(cur_batch_size, max_length)
        return loss, predicted_tags

def do_pass(data, token_to_id, tag_to_id, id_to_tag, id_to_token, expressions, train):
    model, optimizer = expressions
    loss = 0
    match = 0
    total = 0
    start = 0
    while start < len(data):
        batch = data[start : min(start + BATCH_SIZE, len(data))]
        batch.sort(key = lambda x: -len(x[0]))
        cur_batch_size = len(batch)
        start += BATCH_SIZE
        if start % 4000 == 0:
            print(loss, match / total)

        max_length = len(batch[0][0])
        lengths = [len(v[0]) for v in batch]
        xt = torch.zeros((cur_batch_size, max_length)).long() # .long() casts the type from Tensor to LongTensor
        yt = torch.zeros((cur_batch_size, max_length)).long()
        for n, (tokens, tags) in enumerate(batch):
            # This caused initalisation errors... unable to work out why
###            tokens += [PAD] * (max_length - len(tokens))
###            tags += [PAD] * (max_length - len(tags))
            tokens = [token_to_id.get(simplify_token(t), token_to_id[UNK]) for t in tokens]
            tags = [tag_to_id[t] for t in tags]
            xt[n, :len(tokens)] = torch.LongTensor(tokens)
            yt[n, :len(tags)] = torch.LongTensor(tags)

        batch_loss, output = model(xt, yt, lengths, cur_batch_size)

        if train:
            batch_loss.backward()
            optimizer.step()
            model.zero_grad()
            loss += batch_loss.item()
        auto = output.cpu().data.numpy()

        for (_, g), a in zip(batch, auto):
            g = [tag_to_id[t] for t in g]
            for gt, at in zip(g, a):
                if gt != 0:
                    total += 1
                    if gt == at:
                        match += 1
       
    return loss, total, match

if __name__ == '__main__':
    main()