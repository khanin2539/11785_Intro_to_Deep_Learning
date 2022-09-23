


import os
from pickletools import optimize
from re import X
import sys
import pandas as pd
import numpy as np
import Levenshtein as lev
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.utils as utils
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
import datetime
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

cuda = torch.cuda.is_available()

print(cuda, sys.version)

device = torch.device("cuda" if cuda else "cpu")

np.random.seed(11785)
torch.manual_seed(11785)

# # The labels of the dataset contain letters in LETTER_LIST.
# # You should use this to convert the letters to the corresponding indices
# # and train your model with numerical labels.
LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']


# # In[ ]:


# def create_dictionaries(letter_list):
#     '''
#     Create dictionaries for letter2index and index2letter transformations
#     based on LETTER_LIST

#     Args:
#         letter_list: LETTER_LIST

#     Return:
#         letter2index: Dictionary mapping from letters to indices
#         index2letter: Dictionary mapping from indices to letters
#     '''
    

#     letter2index = {letter: i for i, letter in enumerate(letter_list)}
#     index2letter = {i:letter for i, letter in enumerate(letter_list)}
#     return letter2index, index2letter

def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = {letter: i for i, letter in enumerate(letter_list)}
    index2letter = {i:letter for i, letter in enumerate(letter_list)}
    return letter2index, index2letter


def transform_index_to_letter(batch_indices):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''

    # transcripts = np.vectorize(self.index2letter.get)(batch_indices)
    # return transcripts.tolist()

    # transcripts = [LETTER_LIST[index] for index in batch_indices]
    transcripts = []
    # TODO
    for idx in batch_indices:
        transcripts.append( LETTER_LIST[idx])
    return transcripts
  

def transform_index_to_letter(batch_indices):
  
  letters = []
  for row in batch_indices:
    letter = ''
    for i in row:
      if i == letter2index['<eos>']:
        break
      letter += index2letter[i]
    letters.append(letter)
  return letters
        
# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)


# # Dataset and Dataloading (TODO)
# 
# You will need to implement the Dataset class by your own. You can implement it similar to HW3P2. However, you are welcomed to do it your own way if it is more comfortable or efficient.
# 
# Note that you need to use LETTER_LIST to convert the transcript into numerical labels for the model.
# 
# 
# Example of raw transcript:
# 
#     ['<sos>', 'N', 'O', 'R', 'T', 'H', 'A', 'N', 'G', 'E', 'R', ' ','A', 'B', 'B', 'E', 'Y', '<eos>']
# 
# Example of converted transcript ready to process for the model:
# 
#     [0, 14, 15, 18, 20, 8, 1, 14, 7, 5, 18, 28, 1, 2, 2, 5, 25, 29]
# 

# In[ ]:


class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, shuffle=True, partition= "train"):
        self.X_dir =  data_path  + partition + "/mfcc/" # TODO: get mfcc directory path
        self.Y_dir =  data_path  + partition +"/transcript/" # TODO: get transcript path
        self.X_files =  os.listdir(self.X_dir)# TODO: list files in the mfcc directory
        self.Y_files =  os.listdir(self.Y_dir)# TODO: list files in the transcript directory
        if shuffle == True:
            XY_names = list(zip(self.X_files, self.Y_files))
            random.shuffle(XY_names)
            self.X_files, self.Y_files = zip(*XY_names)

        self.LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']


    def __len__(self):
        # TODO
        return len(self.X_files)

    def __getitem__(self, ind):
        X_path = self.X_dir + self.X_files[ind]
        Y_path = self.Y_dir + self.Y_files[ind]
        X = np.load(X_path)
        X = (X - X.mean(axis=0))/np.std(X, axis=0)
        X =  torch.from_numpy(X)
        path = np.load(Y_path)
        
        label = [letter2index[yy] for yy in path]
        Y = np.array(label)
        
        Y =  torch.from_numpy(Y).type(torch.LongTensor)

        return X, Y
    
    def collate_fn(batch):

        # TODO
        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        batch_x_pad = pad_sequence(batch_x, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(x) for x in batch_x] # TODO: Get original lengths of the sequence before padding

        batch_y_pad = pad_sequence(batch_y, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [len(y) for y in batch_y] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)

class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order="test_order.csv"):

        self.X_dir = data_path  + "test" + "/mfcc/"# TODO: Load the npy files from test_order.csv and append into a list
        # self.X = os.listdir(self.X_dir)
        self.X = os.listdir(self.X_dir)
        # print(len(self.X))
        if test_order:
        
            self.X = list(pd.read_csv(test_order).file)# TODO: open test_order.csv as a list
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        # TODO
        x = []
        
        X_path = self.X_dir + self.X[ind]
        
        X = np.load(X_path)
        X = (X - X.mean(axis=0))/np.std(X, axis=0)
        X =  torch.from_numpy(X)
        # print(X)
        return X
    
    def collate_fn(batch):
        # TODO
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(x) for x in batch_x]# TODO: Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x)


# In[ ]:
def cal_lev_dist(predictions, Ys):
  lev_distance = np.array([])
  for pred, y in zip(predictions, Ys):
    pred = pred.replace('<sos>', '')
    y = y.replace('<sos>', '')
   
    distance = lev.distance(pred, y)
    lev_distance = np.append(lev_distance, distance)
  return lev_distance.mean()



batch_size = 128

root = './hw4p2_student_data/hw4p2_student_data/'

train_data = LibriSamples(root, 'train')
val_data = LibriSamples(root, partition = 'dev')
test_data = LibriSamplesTest(root)


train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size =batch_size,   collate_fn = LibriSamples.collate_fn,num_workers=0, pin_memory=True)
val_loader =  torch.utils.data.DataLoader(val_data, shuffle=True, batch_size =batch_size,   collate_fn = LibriSamples.collate_fn,num_workers=0, pin_memory=True)# TODO: Define the val loader. Remember to pass in a parameter (function) for the collate_fn argument 
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size =batch_size,collate_fn = LibriSamplesTest.collate_fn ,num_workers=0, pin_memory=True)# TODO: Define the test loader. Remember to pass in a parameter (function) for the collate_fn argument 



print("Batch size: ", batch_size)
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))


# In[ ]:


# test code for checking shapes
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, len(ly))
    print(y[0]) # desired 
    break



class Batchnorm(nn.Module):
    def __init__(self, encoder_hidden_dim):
        super(pBLSTM, self).__init__()
        self.batchnorm =  nn.BatchNorm1d(encoder_hidden_dim*2)
        self.drop_layer = LockedDropOut()
        
        self.pBLSTMs = nn.Sequential(
            nn.BatchNorm1d(encoder_hidden_dim*4),
            LockedDropOut()
        )

# reference: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html
# class LockedDropOut(nn.Module):
#     def __init__(self, p):
#         super().__init__()
#         self.p = p
#     def forward(self, x):
#         """
#         :param x: (T,B,C) or (B,T,C); T dimension is specified
#         :param p: probability
#         :return:
#         """
#         if not self.training or self.p==0:
#             return x
#         x, x_lens = pad_packed_sequence(x, batch_first = True)
#         mask = torch.zeros(size=(x.size(0), 1, x.size(2)), requires_grad=False)
#         mask = mask.to(device)
#         mask = mask.bernoulli_(1-self.p)
#         # mask = 1/(1-self.p)*mask
#         mask = mask.div_(1 - self.p)
#         mask = mask.expand_as(x)
#         x *=mask
#         x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
#         return x

# reference: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html
class LockedDropout(nn.Module):

    def __init__(self, p=0.3):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or self.p==0:
            return x
        
        x, xlens = pad_packed_sequence(x, batch_first=True)
        xlens = torch.LongTensor(xlens)
        x = x.permute(1,0,2)
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        mask = mask.permute(1,0,2)
        x = x.permute(1,0,2) 
        x*=mask
        output = pack_padded_sequence(x, lengths=xlens, batch_first=True, enforce_sorted=False)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'




class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed
    2. Truncate the input length dimension by concatenating feature dimension
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,bidirectional=True, batch_first=True , num_layers=1)
        


    def forward(self, x):
        x, x_lens = pad_packed_sequence(x, batch_first=True) 
        x_lens = x_lens.to(device)
        x_lens = [length//2 for length in x_lens]
        if x.shape[1] %2 !=0:
            x = x[:, :-1] 
        else:
            x= x

        # chop off extra odd/even sequence
        x = x.contiguous().view(x.size(0), x.size(1) // 2, x.size(2) * 2)
        x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        out, _ = self.blstm(x)
        return out


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)# fill this out)
        
        # Define the blocks of pBLSTMs
        # Dimensions should be chosen carefully
        # Hint: Bidirectionality, truncation...
        self.pBLSTMs = nn.Sequential(
            # LockedDropOut(0.5), 
            pBLSTM(encoder_hidden_dim*4, encoder_hidden_dim),
            # nn.BatchNorm1d(encoder_hidden_dim*2),
            LockedDropout(0.2), 
            pBLSTM(encoder_hidden_dim*4, encoder_hidden_dim),
            # nn.BatchNorm1d(encoder_hidden_dim*2),
            LockedDropout(0.2), 
            pBLSTM(encoder_hidden_dim*4, encoder_hidden_dim), 
            # nn.Dropout(p=0.5)/
            LockedDropout(0.2)
        )
         
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? 
        
        self.key_network = nn.Linear(encoder_hidden_dim*2, key_value_size, bias=True)
        self.value_network = nn.Linear(encoder_hidden_dim*2, key_value_size, bias=True)

    def forward(self, x, x_len):
        """
        1. Pack your input and pass it through the first LSTM layer (no truncation)
        2. Pass it through the pyramidal LSTM layer
        3. Pad your input back to (B, T, *) or (T, B, *) shape
        4. Output Key, Value, and truncated input lens

        Key and value could be
            (i) Concatenated hidden vectors from all time steps (key == value).
            (ii) Linear projections of the output from the last pBLSTM network.
                If you choose this way, you can use the final output of
                your pBLSTM network.
        """
        # print(x_len.shape)

        x_packed = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(x_packed)
        outputs = self.pBLSTMs(outputs)
        out, encoder_len  = pad_packed_sequence(outputs, batch_first=True) 
        key = self.key_network(out)
        val = self.value_network(out)
        return key, val, encoder_len



def plot_attention(attention):
    # utility function for debugging
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    plt.show()

class Attention(nn.Module):
    '''
    Attention is calculated using key and value from encoder and query from decoder.
    Here are different ways to compute attention and context:
    1. Dot-product attention
        energy = bmm(key, query) 
        # Optional: Scaled dot-product by normalizing with sqrt key dimension
        # Check "attention is all you need" Section 3.2.1
    * 1st way is what most TAs are comfortable with, but if you want to explore...
    2. Cosine attention
        energy = cosine(query, key) # almost the same as dot-product xD 
    3. Bi-linear attention
        W = Linear transformation (learnable parameter): d_k -> d_q
        energy = bmm(key @ W, query)
    4. Multi-layer perceptron
        # Check "Neural Machine Translation and Sequence-to-sequence Models: A Tutorial" Section 8.4
    
    After obtaining unnormalized attention weights (energy), compute and return attention and context, i.e.,
    energy = mask(energy) # mask out padded elements with big negative number (e.g. -1e9)
    attention = softmax(energy)
    context = bmm(attention, value)

    5. Multi-Head Attention
        # Check "attention is all you need" Section 3.2.2
        h = Number of heads
        W_Q, W_K, W_V: Weight matrix for Q, K, V (h of them in total)
        W_O: d_v -> d_v

        Reshape K: (B, T, d_k)
        to (B, T, h, d_k // h) and transpose to (B, h, T, d_k // h)
        Reshape V: (B, T, d_v)
        to (B, T, h, d_v // h) and transpose to (B, h, T, d_v // h)
        Reshape Q: (B, d_q)
        to (B, h, d_q // h)

        energy = Q @ K^T
        energy = mask(energy)
        attention = softmax(energy)
        multi_head = attention @ V
        multi_head = multi_head reshaped to (B, d_v)
        context = multi_head @ W_O
    '''
    def __init__(self):
        super(Attention, self).__init__()
        # Optional: dropout

    def forward(self, query, key, value, mask):
        """
        input:
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            query: (batch_size, d_q)
        * Hint: d_k == d_v == d_q is often true if you use linear projections
        return:
            context: (batch_size, key_val_dim)
        
        """

        # energy = torch.bmm(key, query[:, :, None]).squeeze(2)
        energy = torch.bmm(key, query.unsqueeze(-1)).squeeze(-1)
        
        # mask =  torch.arange(key.size(1)).unsqueeze(0) >= mask.unsqueeze(1) # (1, T) >= (B, 1) -> (N, T_max)
        # mask = lens.to(device)
        # energy.masked_fill_(mask, -1e9) # mask out padded elements with big negative number (e.g. -1e9)
        energy.masked_fill_(mask,  -1e9)
        # print(energy)
        # attention = nn.functional.softmax(energy, dim=1)
        attention = nn.functional.softmax(energy/(key.shape[-1]**0.5), dim=1)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1)
        return context, attention
        # we return attention weights for plotting (for debugging)


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        # Hint: Be careful with the padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=letter2index['<eos>'])
        # The number of cells is defined based on the paper
        
        self.lstm1 = nn.LSTMCell(input_size=embed_dim+key_value_size, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)
    
        self.attention = Attention()     
        self.vocab_size = vocab_size

        # Optional: Weight-tying
        self.character_prob = nn.Linear(2*key_value_size, vocab_size, bias=True) # fill this out) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight
        # self.init_weights()

    def forward(self, key, value, encoder_len, y=None, mode='train', teacher_forcing_rate=None):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, text_len) - Batch input of text with text_length
            mode: Train or eval mode for teacher forcing
        Return:
            predictions: the character perdiction probability 
        '''

        B, key_seq_max_len, key_value_size = key.shape

        if mode == 'train':
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600

        # TODO: Create the attention mask here (outside the for loop rather than inside) to aviod repetition
        mask = torch.arange(key_seq_max_len).unsqueeze(0) >= torch.as_tensor(encoder_len).unsqueeze(1)
        mask = mask.to(device)
        
        predictions = []
        # This is the first input to the decoder
        # What should the fill_value be? -> 1 because it's prediction?
        prediction = torch.zeros(B, 1).to(device)
       
        hidden_states = [None, None] 
        
        # TODO: Initialize the context
        
        context =  torch.zeros(B, key_value_size).to(device)

        attention_plot = [] # this is for debugging
        # print(random_rate)
        for i in range(max_len):
            if mode == 'train':
                # TODO: Implement Teacher Forcing -> Use binomial distribution for teacher forcing

                teacher_forcing = np.sum(np.random.binomial(n=1, p=teacher_forcing_rate))
                if teacher_forcing == 1:
                    if i == 0:
                        start_char =char_embeddings[:, 0]
                        char_embed = torch.zeros_like(start_char).to(device)
                    else:
                        char_embed = char_embeddings[:, i-1]
                else: # not teacher forcing
                    char_embed = self.embedding(prediction.argmax(dim=-1))
                    
            else: # validate
                if i==0:
                    start_char = torch.zeros(B, dtype=torch.long).fill_(letter2index['<sos>']).to(device)
                    char_embed = self.embedding(start_char) # embedding of the previous prediction
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))


            # what vectors should be concatenated as a context?
            y_context = torch.cat([char_embed, context], dim=1)
            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0]) # Input of shape batch × input dimension; A tuple of LSTM hidden states of shape batch x hidden dimensions.

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1]) # gives out hidden states
            # What then is the query?
            query = hidden_states[1][0] # fill this out
            
            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query, key, value, mask)
            # We store the first attention of this batch for debugging
            attention_plot.append(attention[0].detach().cpu())
            
            # What should be concatenated as the output context?
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            # store predictions
            predictions.append(prediction.unsqueeze(1))
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim)# fill this out)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, key_value_size)# fill this out)

    def forward(self, x, x_len, y=None, mode='train',  teacher_forcing_rate=None):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode,  teacher_forcing_rate=teacher_forcing_rate)
        return predictions


# In[ ]:


model = Seq2Seq(input_dim = 13, vocab_size=len(LETTER_LIST), encoder_hidden_dim=256, decoder_hidden_dim=256, embed_dim=256).to(device)

model = model.to(device)
print(model)


# # Training




optimizer = optim.Adam(model.parameters(), lr = 0.002)# fill this out)
scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=1, verbose=True, threshold=1e-2)# fill this out
criterion = nn.CrossEntropyLoss(reduction='none')


# In[ ]:
def generate_mask(lens):
    lens = torch.tensor(lens).to(device)
    max_len = torch.max(lens)

    mask = (torch.arange(0, max_len).repeat(lens.size(0), 1).to(device) <=lens[:, None].expand(-1, max_len)).int()
    return mask

def train(model, train_loader, criterion, optimizer, mode, scaler, epoch, teacher_forcing_rate):
    model.train()
    running_loss =0
    
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    batch = len(train_loader)
    # 0) Iterate through your data loader
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y, x_len, y_len = data
        # print(x.shape, y.shape, lx.shape, len(ly))
        # 1) Send the inputs to the device
        x = x.to(device)
        y = y.to(device)
        
        # 2) Pass your inputs, and length of speech into the model.
        predictions = model(x, x_len, y, mode=mode, teacher_forcing_rate =teacher_forcing_rate)
        loss = criterion(predictions.view(-1, predictions.shape[-1]), y.view(-1))
        # loss = criterion(predictions.view(-1, predictions.size(2)), y.view(-1))# fill this out)
        # 3) Generate a mask based on target length. This is to mark padded elements
        # so that we can exclude them from computing loss.
        # Ensure that the mask is on the device and is the correct shape.



        mask = generate_mask(y_len).to(device)# fill this out
            
        # 4) Make sure you have the correct shape of predictions when putting into criterion
       
        # Use the mask you defined above to compute the average loss
        masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)# fill this out
        
        
        # 5) backprop
        masked_loss.backward()
        # scheduler.step()
        optimizer.step()
        running_loss += masked_loss.item()
        # running_loss = running_loss / (i + 1)
        
        batch_bar.set_postfix(
            
            loss="{:.04f}".format(running_loss/ (i + 1)),
            # dis ="{}".format(float(distance))
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
            
            )

        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar
    model_name = 'attention_baseline_add_tf_bs_128_1.pt'
    torch.save(model, model_name)
    print("Epoch {}, Train Loss {:.04f}".format(
    epoch + 1,
    float(running_loss / len(train_loader))))
    # return  teacher_forcing_rate, float(optimizer.param_groups[0]['lr'])
    




def val(model, criterion, valid_loader, mode, epoch):
    with torch.no_grad():
        model.eval()
        running_loss = 0
        runningDist = 0
        num_seq = 0
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc='Val') 
        
        for i, data in enumerate(val_loader):
            x, y, x_len, y_len = data
            # print(x.shape, y.shape, lx.shape, len(ly))
            # 1) Send the inputs to the device
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
            # 2) Pass your inputs, and length of speech into the model.
                predictions = model(x, x_len, y, mode=mode)
            
            # 3) Generate a mask based on target length. This is to mark padded elements
            # so that we can exclude them from computing loss.
            # Ensure that the mask is on the device and is the correct shape.
            mask = generate_mask(y_len).to(device)# fill this out
                
            # 4) Make sure you have the correct shape of predictions when putting into criterion
            
            pred_sents = transform_index_to_letter(torch.argmax(predictions, dim=-1).to('cpu').numpy())
            target_sents = transform_index_to_letter(y.to('cpu').numpy())
            print(pred_sents)
            print(target_sents)
            

            runningDist += cal_lev_dist(pred_sents, target_sents)
            num_seq += len(pred_sents)

           
            batch_bar.set_postfix(
                
                loss="{:.04f}".format(float((runningDist) / (i + 1))),
                # dis ="{}".format(float(distance))
                lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
                
                )

            batch_bar.update() # Update tqdm bar
        batch_bar.close() # You need this to close the tqdm bar
        print("Epoch {}, runningDist/num_seq {:.04f}".format(
        epoch + 1,
        float(runningDist/(i + 1))))
        # return runningDist / (i + 1)


# # In[ ]:


# TODO: Define your model and put it on the device here
# ...
# model = torch.load('attention_baseline_add_tf_bs_32.pt')
scaler = torch.cuda.amp.GradScaler()
n_epochs = 50
# optimizer = optim.Adam(model.parameters(), # fill this out)
# Make sure you understand the implication of setting reduction = 'none'
criterion = nn.CrossEntropyLoss(reduction='none')
mode = 'train'
# model = torch.load('attention_baseline_add_tf_bs_128.pt')
tf_rate = 0.7
for epoch in range(n_epochs):

    train(model, train_loader, criterion, optimizer, mode, scaler=scaler, epoch=epoch, teacher_forcing_rate=tf_rate)
    val(model, criterion, val_loader, model, epoch=epoch)
    if epoch>20 and tf_rate>0.35 and not epoch%5:
        tf_rate-=0.05
    

   



def test(model, criterion, test_loader, mode):
    torch.cuda.empty_cache()
    model.eval()
    letters = []
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc='Test')
    for i, data in enumerate(test_loader):
        x, x_len = data
        x = x.to(device)
        with torch.no_grad():
          predictions = model(x, x_len, y=None, mode=mode) 
        pred_text = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy())
       
        letters.extend(pred_text)
        
        batch_bar.update() 
      
    batch_bar.close() 

    # remove <eos> in the prediction
    letters = [l.replace('<eos>', '') for l in letters]
    # remove <sos> in the prediction
    letters = [l.replace('<sos>', '') for l in letters]
    
    return letters

results = test(model, criterion, test_loader, mode="Test")
indexes = range(len(results))
import csv
with open("hw4p2_submission.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows([['id', 'predictions']])
    for i, s in zip(indexes, results):
        writer.writerow([i, s])
    


'''
Debugging suggestions from Eason, a TA from previous semesters:

(1) Decrease your batch_size to 2 and print out the value and shape of all intermediate variables to check if they satisfy the expectation
(2) Be super careful about the LR, don't make it too high. Too large LR would lead to divergence and your attention plot will never make sense
(3) Make sure you have correctly handled the situation for time_step = 0 when teacher forcing

(1) is super important and is the most efficient way for debugging. 
'''
'''
Tips for passing A from B (from easy to hard):
** You need to implement all of these yourself without utilizing any library **
(1) Increase model capacity. E.g. increase num_layer of lstm
(2) LR and Teacher Forcing are also very important, you can tune them or their scheduler as well. Do NOT change lr or tf during the warm-up stage!
(3) Weight tying
(4) Locked Dropout - insert between the plstm layers
(5) Pre-training decoder or train an LM to help make predictions
(5) Pre-training decoder to speed up the convergence: 
    disable your encoder and only train the decoder like train a language model
(6) Better weight initialization technique
(7) Batch Norm between plstm. You definitely can try other positions as well
(8) Data Augmentation. Time-masking, frequency masking
(9) Weight smoothing (avg the last few epoch's weight)
(10) You can try CNN + Maxpooling (Avg). Some students replace the entire plstm blocks with it and some just combine them together.
(11) Beam Search
'''



