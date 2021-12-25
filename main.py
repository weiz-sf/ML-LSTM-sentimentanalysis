# -*- coding: utf-8 -*-
!pip install spacy
!python -m spacy download en_core_web_sm
!pip install torchtext==0.4.0

import torch
import torchtext
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import torch.optim as optim
import time
import warnings
warnings.filterwarnings('ignore')

# TEXT = data.Field(include_lengths=True)

# If you want to use English tokenizer from SpaCy, you need to install SpaCy and download its English model:
# pip install spacy
# python -m spacy download en_core_web_sm
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)

LABEL = data.LabelField(dtype=torch.long)
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train_data)
# Here, you can also use some pre-trained embedding
TEXT.build_vocab(train_data,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size=batch_size, device=device)

print(len(TEXT.vocab))

#Parameter description 
embed_dim = 100
hidden_dim = 256
n_layers = 2
dropout = 0.5
num_epochs = 10
weight_decay = 0
learning_rate = 0.0001
input_dim = len(TEXT.vocab)
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

output_dim = 2
batch_size = 64

import matplotlib.pyplot as plt
import numpy as np

# Create Model Class
# Bidirectional LSTM RNN
class BidirectionLSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers = n_layers, bidirectional = True, dropout = dropout)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):

        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted = False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        hidden_fwd = hidden[-2]
        hidden_bck = hidden[-1]

        hidden = torch.cat((hidden_fwd, hidden_bck), dim = 1)

        prediction = self.fc(self.dropout(hidden))

        return prediction

model = BidirectionLSTM(input_dim, embed_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx)
def initialize_parameters(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif 'weight_hh' in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif 'bias' in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(initialize_parameters)

# Embedding layer
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Zero the initial weight
model.embedding.weight.data[pad_idx] = torch.zeros(embed_dim)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss() # In binary classifcation, CrossEntropyLoss become Binary Cross Entropy Loss

# Train, Test and Find Accuracy 
model = model.to(device)
criterion = criterion.to(device)

def calculate_accuracy(predictions, labels):
    top_predictions = predictions.argmax(1, keepdim = True)
    correct = top_predictions.eq(labels.view_as(top_predictions)).sum()
    accuracy = correct.float() / labels.shape[0]
    return accuracy

def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for text_arr, labels in iterator:
        
        labels = labels.to(device)
        text = text_arr[0].to(device)

        optimizer.zero_grad()
        
        predictions = model(text, text_arr[1].cpu())
        
        loss = criterion(predictions, labels)
        
        acc = calculate_accuracy(predictions, labels)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test_accuracy(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for text_arr, labels in iterator:

            labels = labels.to(device)
            text = text_arr[0].to(device)
            
            predictions = model(text, text_arr[1].cpu())
            
            loss = criterion(predictions, labels)
            
            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

avg_loss_list=[]
for epoch in range(num_epochs):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = test_accuracy(model, valid_iterator, criterion, device)
    avg_loss_list.append(train_loss)
    print(f'Epoch: {epoch+1}')
    print(f'\tAverage Batch Loss: {train_loss:.3f} & Train Accuracy: {train_acc*100:.2f}%')
    print(f'\tAverage Val. Loss: {valid_loss:.3f} &  Validation Accuracy: {valid_acc*100:.2f}%')

test_loss, test_acc = test_accuracy(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} & Test Accuracy: {test_acc*100:.2f}%')


plt.title('LSTM')
plt.xlabel('epochs')
plt.ylabel('average batch loss')
plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), avg_loss_list)
