

# %%
print('hello')

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score
import gc
import zipfile
import pandas as pd
from tqdm import tqdm
import os
import datetime
from phonemes import PHONEME_MAP

# imports for decoding and distance calculation
import ctcdecode
import Levenshtein
from ctcdecode import CTCBeamDecoder
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# %%
import random


PHONEME_MAP = [
    " ",
    ".", #SIL
    "a", #AA
    "A", #AE
    "h", #AH
    "o", #AO
    "w", #AW
    "y", #AY
    "b", #B
    "c", #CH
    "d", #D
    "D", #DH
    "e", #EH
    "r", #ER
    "E", #EY
    "f", #F
    "g", #G
    "H", #H
    "i", #IH 
    "I", #IY
    "j", #JH
    "k", #K
    "l", #L
    "m", #M
    "n", #N
    "N", #NG
    "O", #OW
    "Y", #OY
    "p", #P 
    "R", #R
    "s", #S
    "S", #SH
    "t", #T
    "T", #TH
    "u", #UH
    "U", #UW
    "v", #V
    "W", #W
    "?", #Y
    "z", #Z
    "Z" #ZH
]

# %%
arr = [1, 2, 4, 5, 6, 7, 8, 3]
arr[1:-2]

# %%
# This cell is where your actual TODOs start
# You will need to implement the Dataset class by your own. You may also implement it similar to HW1P2 (dont require context)
# The steps for implementation given below are how we have implemented it.
# However, you are welcomed to do it your own way if it is more comfortable or efficient. 

class LibriSamples(torch.utils.data.Dataset):

    def __init__(self, data_path, shuffle=True, partition= "train"): # You can use partition to specify train or dev
        # print(data_path)
        self.X_dir =  data_path  + partition + "/mfcc/" # TODO: get mfcc directory path
        self.Y_dir =  data_path  + partition +"/transcript/" # TODO: get transcript path

    
    # def __getitem__(self, idx):
    #     return torch.from_numpy(self.dataX[idx]).float(), torch.from_numpy(self.dataY[idx] + 1 if self.dataY is not None else np.array([-1])).int() # add 1 to label to account for blank
    
    # def __len__(self):
    #     return len(self.dataX)

        self.X_files =  os.listdir(self.X_dir)# TODO: list files in the mfcc directory
        self.Y_files =  os.listdir(self.Y_dir)# TODO: list files in the transcript directory
        if shuffle == True:
            XY_names = list(zip(self.X_files, self.Y_files))
            random.shuffle(XY_names)
            self.X_files, self.Y_files = zip(*XY_names)

        # TODO: store PHONEMES from phonemes.py inside the class. phonemes.py will be downloaded from kaggle.
        # You may wish to store PHONEMES as a class attribute or a global variable as well.
        self.PHONEMES = ["", 'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH']

        assert(len(self.X_files) == len(self.Y_files))

        # pass

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):
    
        X = []# TODO: Load the mfcc npy file at the specified index ind in the directory

        Y = []# TODO: Load the corresponding transcripts
        # for j in range(len(self.X_files)):
        X_path = self.X_dir + self.X_files[ind]
        Y_path = self.Y_dir + self.Y_files[ind]
            # print(Y_path)
        path = np.load(Y_path)
        path = (path[1:-1])
            
        label = [self.PHONEMES.index(yy) for yy in path]
        # print(label)
        
        X = np.load(X_path)
        X =  torch.from_numpy(X)
            
            # X.append(X_data)
        Y = np.array(label)
        # print('X', type(X))
    
        # Remember, the transcripts are a sequence of phonemes. Eg. np.array(['<sos>', 'B', 'IH', 'K', 'SH', 'AA', '<eos>'])
        # You need to convert these into a sequence of Long tensors
        # Tip: You may need to use self.PHONEMES
        # Remember, PHONEMES or PHONEME_MAP do not have '<sos>' or '<eos>' but the transcripts have them. 
        # You need to remove '<sos>' and '<eos>' from the trancripts. 
        # Inefficient way is to use a for loop for this. Efficient way is to think that '<sos>' occurs at the start and '<eos>' occurs at the end.
        
        Yy =  torch.from_numpy(Y).long() # TODO: Convert sequence of  phonemes into sequence of Long tensors
        # print('Y', Yy.shape)
        # print('Y', Yy)

        return X, Yy
    
    
    def collate_fn(batch):
        batch_x = [x for x,y in batch]
        batch_y = [y for x,y in batch]

        batch_x_pad = pad_sequence(batch_x, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(x) for x in batch_x] # TODO: Get original lengths of the sequence before padding

        batch_y_pad = pad_sequence(batch_y, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [len(y) for y in batch_y] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)
        # batch_x_pad = pad_sequence(batch_x)# TODO: pad the sequence with pad_sequence (already imported)
        # lengths_x =  [len(seq[0]) for seq in batch]# TODO: Get original lengths of the sequence before padding

        # batch_y_pad = pad_sequence(batch_y) # TODO: pad the sequence with pad_sequence (already imported)
        # lengths_y = [len(seq[1]) for seq in batch]# TODO: Get original lengths of the sequence before padding

        # return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)


# You can either try to combine test data in the previous class or write a new Dataset class for test data
class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order="test_order.csv"): # test_order is the csv similar to what you used in hw1
        
        self.X_dir = data_path  + "test" + "/mfcc/"# TODO: Load the npy files from test_order.csv and append into a list
        # self.X = os.listdir(self.X_dir)
        self.X = os.listdir(self.X_dir)
        print(len(self.X))
        if test_order:
        
            self.X = list(pd.read_csv(test_order).file)# TODO: open test_order.csv as a list
        # You can load the files here or save the paths here and load inside __getitem__ like the previous class
        # print(self.X)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        # TODOs: Need to return only X because this is the test dataset
        # x = self.X[ind]
        x = []
        
        X_path = self.X_dir + self.X[ind]
        print(X_path)
            #Y_path = self.Y_dir + self.Y_names[j]
            
            #label = [self.PHONEMES.index(yy) for yy in np.load(Y_path)][1:-1]

        
        # X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
        X = np.load(X_path)
        X =  torch.from_numpy(X)
        # print(X)
        return X
    
    def collate_fn(batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True)# TODO: pad the sequence with pad_sequence (already imported)
        lengths_x = [len(x) for x in batch_x]# TODO: Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x)


# %%
batch_size = 128

root = './hw3p2_student_data/'# TODO: Where your hw3p2_student_data folder is

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

# %%
# Optional
# Test code for checking shapes and return arguments of the train and val loaders
# for data in val_loader:
#     x, y, lx, ly = data # if you face an error saying "Cannot unpack", then you are not passing the collate_fn argument
#     print(x.shape, y.shape, lx.shape, ly.shape)
#     break

# %% [markdown]
# # Model Configuration (TODO)



class Network(nn.Module):

    def __init__(self, num_classes=41, input_size=13, hidden_size=256, num_layers=1): # You can add any extra arguments as you wish

        super(Network, self).__init__()
        # input_dim = 13

        self.input_size = input_size

        # Embedding layer converts the raw input into features which may (or may not) help the LSTM to learn better 
        # For the very low cut-off you dont require an embedding layer. You can pass the input directly to the  LSTM
        # self.embedding = nn.Conv1d(in_channels=input_dim, out_channels=5, kernel_size=3, padding=1, stride=1)
        
        # TODO: # Create a single layer, uni-directional LSTM with hidden_size = 256
        # Use nn.LSTM() Make sure that you give in the proper arguments as given in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            bias=True, batch_first=True, dropout=0, bidirectional=False, proj_size=0)

        self.classification = nn.Linear(hidden_size, num_classes) # TODO: Create a single classification layer using nn.Linear()
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x, lx): # TODO: You need to pass atleast 1 more parameter apart from self and x

        # x is returned from the dataloader. So it is assumed to be padded with the help of the collate_fn
        # TODO: Pack the input with pack_padded_sequence. Look at the parameters it requires
        # torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
        packed_input = pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)

        output, (h_n, c_n) = self.lstm(packed_input) # TODO: Pass packed input to self.lstm
        # output: tensor of shape (L, D * H_{out})(L,D∗H out) for unbatched input, (L, N, D * H_{out})
        # (L,N,D∗Hout) when batch_first=False or (N, L, D * H_out)(N,L,D∗Hout) when batch_first=True 
        # containing the output features (h_t) from the last layer of the LSTM, for each t. 
        # If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.
        # h_n: tensor of shape (D * num_layers, H_{out})(D∗num_layers,Hout) for unbatched input 
        # or (D * num_layers, N, H_{out})(D∗num_layers,N,Hout) 
        # containing the final hidden state for each element in the sequence.
        # c_n: tensor of shape (D * num_layers, H_{cell})(D∗num_layers,Hcell) for unbatched input or (D * num_layers, N, H_cell)
        # (D∗num_layers,N,Hcell) containing the final cell state for each element in the sequence.
        
        # As you may see from the LSTM docs, LSTM returns 3 vectors. Which one do you need to pass to the next function?
        out, lengths = pad_packed_sequence(output, batch_first=True) # TODO: Need to 'unpack' the LSTM output using pad_packed_sequence

        # out = self.classification(out) # TODO: Pass unpacked LSTM output to the classification layer
        out = self.classification(out).log_softmax(2) # Optional: Do log softmax on the output. Which dimension?

        return out, lengths # TODO: Need to return 2 variables

model = Network().to(device)
# print(model)
# summary(model, x.to(device), lx) # x and lx are from the previous cell



# summary(model, x.to(device), lx) # x and lx are from the previous cell

# %% [markdown]
# # Training Configuration (TODO)

# %%
criterion = nn.CTCLoss() # TODO: What loss do you need for sequence to sequence models? 
# Do you need to transpose or permute the model output to find out the loss? Read its documentation
optimizer =  torch.optim.Adam(model.parameters(), lr=2e-3)# TODO: Adam works well with LSTM (use lr = 2e-3)
decoder = CTCBeamDecoder(labels=PHONEME_MAP,
                            num_processes=os.cpu_count(), log_probs_input=True) # TODO: Intialize the CTC beam decoder
                
# Check out https://github.com/parlance/ctcdecode for the details on how to implement decoding
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) *20))
# Do you need to give log_probs_input = True or False?

# %%
# this function calculates the Levenshtein distance 

def calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP):

    # h - ouput from the model. Probability distributions at each time step 
    # y - target output sequence - sequence of Long tensors
    # lh, ly - Lengths of output and target
    # decoder - decoder object which was initialized in the previous cell
    # PHONEME_MAP - maps output to a character to find the Levenshtein distance

    # TODO: You may need to transpose or permute h based on how you passed it to the criterion
    # Print out the shapes often to debug
    # h = torch.transpose(h, 1, 0)
    # print(type(h))


    # TODO: call the decoder's decode method and get beam_results and out_len (Read the docs about the decode method's outputs)
    # Input to the decode method will be h and its lengths lh 
    # You need to pass lh for the 'seq_lens' parameter. This is not explicitly mentioned in the git repo of ctcdecode.
    beam_results, beam_scores, timesteps, out_len = decoder.decode(h, lh)
    print(beam_results.shape)
    print(out_len.shape)
    print(y.shape, ly.shape)

    # beam_results_y, beam_scores_y, timesteps_y, out_len_y = decoder.decode(y, ly)



    batch_size = beam_results.shape[0]# TODO

    dist = 0

    for i in range(batch_size): # Loop through each element in the batch

        # h_sliced = # TODO: Get the output as a sequence of numbers from beam_results
        # Remember that h is padded to the max sequence length and lh contains lengths of individual sequences
        # Same goes for beam_results and out_lens
        # You do not require the padded portion of beam_results - you need to slice it with out_lens 
        # If it is confusing, print out the shapes of all the variables and try to understand
        # h_string  = []
        # y_string = []
        # if out_len[i][0] != 0:
        h_sliced = beam_results[i][0][:out_len[i][0]]
        h_string = "".join([PHONEME_MAP[x] for x in h_sliced])
           
        y_sliced = y[i][:ly[i]]
        
        # if out_len_y[i][0] != 0:
        y_string = "".join([PHONEME_MAP[y] for y in y_sliced])# TODO: Do the same for y - slice off the padding with ly
        
        dist += Levenshtein.distance(h_string, y_string)

    dist/=batch_size
    # print(dist)

    return dist

def calculate_h_string(h, lh, decoder, PHONEME_MAP):

    # h - ouput from the model. Probability distributions at each time step 
    # y - target output sequence - sequence of Long tensors
    # lh, ly - Lengths of output and target
    # decoder - decoder object which was initialized in the previous cell
    # PHONEME_MAP - maps output to a character to find the Levenshtein distance

    # TODO: You may need to transpose or permute h based on how you passed it to the criterion
    # Print out the shapes often to debug
    # h = torch.transpose(h, 1, 0)
    # print(type(h))


    # TODO: call the decoder's decode method and get beam_results and out_len (Read the docs about the decode method's outputs)
    # Input to the decode method will be h and its lengths lh 
    # You need to pass lh for the 'seq_lens' parameter. This is not explicitly mentioned in the git repo of ctcdecode.
    beam_results, beam_scores, timesteps, out_len = decoder.decode(h, lh)
    print(beam_results.shape)
    print(out_len.shape)
    

    # beam_results_y, beam_scores_y, timesteps_y, out_len_y = decoder.decode(y, ly)



    batch_size = beam_results.shape[0]# TODO

    dist = 0
    h_string_list = []
    for i in range(batch_size): # Loop through each element in the batch

        # h_sliced = # TODO: Get the output as a sequence of numbers from beam_results
        # Remember that h is padded to the max sequence length and lh contains lengths of individual sequences
        # Same goes for beam_results and out_lens
        # You do not require the padded portion of beam_results - you need to slice it with out_lens 
        # If it is confusing, print out the shapes of all the variables and try to understand
        # h_string  = []
        # y_string = []
        # if out_len[i][0] != 0:
        # h_sliced = beam_results[i][0][:out_len[i][0]]
        h_sliced = h[i][:lh[i]]
        h_string = "".join([PHONEME_MAP[x] for x in h_sliced])
        h_string_list.append(h_string)
    print(h_string_list)
    return h_string_list
           





for i, data in enumerate(train_loader, 0):
    x, y, lx, ly = data
    x = x.float().to(device)
    y = y.long().to(device)
    # lx = lx.long().to(device)
    # ly = ly.long().to(device)

    optimizer.zero_grad()
    h, lh = model(x, lx)
    # print(h.shape, y.shape, lh.shape, ly.shape)
    loss = criterion(h.permute(1,0,2), y, lh, ly)
    # loss.backward()
    # optimizer.step()

    distance = calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP)

    # Write a test code do perform a single forward pass and also compute the Levenshtein distance
    # Make sure that you are able to get this right before going on to the actual training
    # You may encounter a lot of shape errors
    # Printing out the shapes will help in debugging
    # Keep in mind that the Loss which you will use requires the input to be in a different format and the decoder expects it in a different format
    # Make sure to read the corresponding docs about it
    pass

    break # one iteration is enough
epoch_number = 0
for epoch in range(20):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    model.train(True)
    loss_epoch = train_one_epoch(epoch_number, )
    model.train(False)


    running_vloss = 0.0
    for i, data in enumerate(val_loader, 0):
        x, y, lx, ly = data
        x = x.float().to(device)
        y = y.long().to(device)
        h, lh = model(x, lx)
        loss = criterion(h.permute(1,0,2), y, lh, ly)
        # loss.backward()
        # optimizer.step()

        distance = calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP)
        running_vloss += loss.item()

    print('LOSS train {} valid {}'.format(loss_epoch, running_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    epoch_number += 1


scaler = torch.cuda.amp.GradScaler()
# model = torch.load('lstm_early_submission.pt')

for epoch in range(20):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0

    for i, data in enumerate(train_loader):
        x, y, lx, ly = data
        x = x.to(device)
        y = y.to(device)

        # Don't be surprised - we just wrap these two lines to make it work for FP16
        optimizer.zero_grad()
        h, lh = model(x, lx)
        # print(h.shape, y.shape, lh.shape, ly.shape)
        loss = criterion(h.permute(1,0,2), y, lh, ly)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        # scaler.scale(loss).backward() # This is a replacement for loss.backward()
        # scaler.step(optimizer) # This is a replacement for optimizer.step()
        # scaler.update() # This is something added just for FP16

        # scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

        # distance = calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP)
        # print(distance)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            # dis ="{}".format(float(distance))
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
            
            )
    
        
#         # Another couple things you need for FP16. 
#         # loss.backward() # This is a replacement for loss.backward()
#         # optimizer.step(optimizer) # This is a replacement for optimizer.step()
#         # scaler.update() # This is something added just for FP16

#         # scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar
    model_name = 'lstm_early_submission_2.pt'
    torch.save(model, model_name)

    # You can add validation per-epoch here if you would like

    print("Epoch {}, Train Loss {:.04f}".format(
        epoch + 1,
        float(total_loss / len(train_loader))))

eval
model = torch.load('lstm_early_submission_2.pt')
model.eval()
batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
val_total_loss = 0
for i, data in enumerate(val_loader, 0):
    x, y, lx, ly = data
    x = x.to(device)
    y = y.to(device)
    h, lh = model(x, lx)
    loss = criterion(h.permute(1,0,2), y, lh, ly)
    distance = calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP)
    # loss.backward()
    # optimizer.step()
    val_total_loss += float(loss)
    batch_bar.set_postfix(
            
            loss="{:.04f}".format(float(val_total_loss / (i + 1))),
            dis ="{}".format(float(distance))
            
            )

    batch_bar.update()

#test
model = torch.load('lstm_early_submission_2.pt')
model.eval()
pred_y_list = []
batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
val_total_loss = 0
for i, data in enumerate(test_loader):
    x, lx = data
    x = x.to(device)
    # y = y.to(device)
    with torch.no_grad():
        h, lh = model(x, lx)
    # h, lh = model(x, lx)
    
    # loss = criterion(h.permute(1,0,2), y, lh, ly)
    # distance = calculate_levenshtein(h, y, lh, ly, decoder, PHONEME_MAP)
    
    beam_results, beam_scores, timesteps, out_len = decoder.decode(h, lh)
    batch_size = beam_results.shape[0]# TODO
    for i in range(batch_size): # Loop through each element in the batch
        h_sliced = beam_results[i][0][:out_len[i][0]]
        h_string = "".join([PHONEME_MAP[x] for x in h_sliced])
    # h_string =   calculate_h_string(h, lh, decoder, PHONEME_MAP)
        pred_y_list.append(h_string)
    

    batch_bar.update()
batch_bar.close()
with open("hw3p2_early_submission.csv", 'w') as fh:
        fh.write('id,predictions\n')     
        for x in range(len(pred_y_list)):
            fh.write(str(x)+ ',' + str(pred_y_list[x]) + "\n")

    
model.eval()
batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')

pred = []
for i, (x, lx) in enumerate(test_loader):
    x = x.cuda()

    with torch.no_grad():
        h, lh = model(x, lx)
        
    h_sliced = h[i][:lh[i]]
    h_string = ''.join([PHONEME_MAP[h] for h in h_sliced])

    pred.append(h_string)

    batch_bar.update()
batch_bar.close()


#     # Write a test code do perform a single forward pass and also compute the Levenshtein distance
#     # Make sure that you are able to get this right before going on to the actual training
#     # You may encounter a lot of shape errors
#     # Printing out the shapes will help in debugging
#     # Keep in mind that the Loss which you will use requires the input to be in a different format and the decoder expects it in a different format
#     # Make sure to read the corresponding docs about it
#     pass

#     break # one iteration is enough

# # %%
# torch.cuda.empty_cache() # Use this often

# # TODO: Write the model evaluation function if you want to validate after every epoch

# # You are free to write your own code for model evaluation or you can use the code from previous homeworks' starter notebooks
# # However, you will have to make modifications because of the following.
# # (1) The dataloader returns 4 items unlike 2 for hw2p2
# # (2) The model forward returns 2 outputs
# # (3) The loss may require transpose or permuting

# # Note that when you give a higher beam width, decoding will take a longer time to get executed
# # Therefore, it is recommended that you calculate only the val dataset's Levenshtein distance (train not recommended) with a small beam width
# # When you are evaluating on your test set, you may have a higher beam width

# def evaluate(model, criterion, loader, args, calc_cer=False):
#     model.eval()
#     total_loss = 0
#     total_cer = 0
#     score_board = []
#     for batch, (frames, seq_sizes, labels, label_sizes) in enumerate(loader):
#         data, target, dataLens, targetLens = data.cuda(), target.cuda(), dataLens.cuda(), targetLens.cuda()
#         output, dataLens_new = model(data, dataLens)
#         loss = criterion(output,
#                              target,
#                              dataLens_new,
#                              targetLens)
#         running_loss += loss.item()
#         totalSampleCnt += len(data)
        
#                 print("Epoch: {}\tBatch: {}\tTimestamp: {}".format(epoch, batch_idx, time.time() - start_time))
#             torch.cuda.empty_cache()










# # %%
# torch.cuda.empty_cache()

# # TODO: Write the model training code 

# # You are free to write your own code for training or you can use the code from previous homeworks' starter notebooks
# # However, you will have to make modifications because of the following.
# # (1) The dataloader returns 4 items unlike 2 for hw2p2
# # (2) The model forward returns 2 outputs
# # (3) The loss may require transpose or permuting

# # Tip: Implement mixed precision training

# # %% [markdown]
# # # Submit to kaggle (TODO)

# # %%
# # TODO: Write your model evaluation code for the test dataset
# # You can write your own code or use from the previous homewoks' stater notebooks
# # You can't calculate loss here. Why?

# # %%
# # TODO: Generate the csv file


