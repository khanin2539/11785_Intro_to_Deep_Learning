# %%
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd



# %%
# !pip install --upgrade --force-reinstall --no-deps kaggle
#!pip install pandas

# %%
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        in_size = 1313
        layers = [
                        nn.Linear(in_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 40)
        ]
        #self.laysers = nn.Sequential(*layers)
        self.laysers = nn.Sequential(nn.Flatten(), *layers)
        # self.model = nn.Sequential(nn.Flatten(), *layers)
        #self.laysers.apply(self.init_weights)
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    # net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    # net.apply(init_weights)

    def forward(self, A0):
        x = self.laysers(A0)
        
        return x

# %%

class LibriSamples(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="dev-clean", csvpath=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 
        
        # get directory path string
        self.X_dir = data_path  + partition + "/mfcc/"
        self.Y_dir = data_path  + partition +"/transcript/"
        # get actual path
        self.X_names = os.listdir(self.X_dir)
        self.Y_names = os.listdir(self.Y_dir)

        # using a small part of the dataset to debug
        if csvpath:
            subset = self.parse_csv(csvpath)
            self.X_names = [i for i in self.X_names if i in subset]
            self.Y_names = [i for i in self.Y_names if i in subset]

            
        
        if shuffle == True:
            XY_names = list(zip(self.X_names, self.Y_names))
            random.shuffle(XY_names)
            self.X_names, self.Y_names = zip(*XY_names)
        
        assert(len(self.X_names) == len(self.Y_names))
        self.length = len(self.X_names)
        
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[1])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
        
        X, Y = [], []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]
            Y_path = self.Y_dir + self.Y_names[j]
            
            label = [self.PHONEMES.index(yy) for yy in np.load(Y_path)][1:-1]

            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
            Y.append(np.array(label))
            
        X, Y = np.concatenate(X), np.concatenate(Y)
        return X, Y
    
class LibriItems(torch.utils.data.Dataset):
    def __init__(self, X, Y, context = 0):
        assert(X.shape[0] == Y.shape[0])
        
        self.length  = X.shape[0]
        self.context = context

        if context == 0:
            self.X, self.Y = X, Y
        else:
            X = np.pad(X, ((context,context), (0,0)), 'constant', constant_values=(0,0))
            self.X, self.Y = X, Y
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
            yy = self.Y[i]
            
        else:
            xx = self.X[i:(i + 2*self.context + 1)].flatten()
            yy = self.Y[i]
        #print(xx.shape)
        return xx, yy


# %%

class LibriSamples_eval(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="test-clean", csvpath="test_order.csv"):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 
        
        self.X_dir = data_path  + partition + "/mfcc/"
        #self.Y_dir = data_path  + partition +"/transcript/"
        print(self.X_dir)
        self.X_names = os.listdir(self.X_dir)
        #self.Y_names = os.listdir(self.Y_dir)

        # using a small part of the dataset to debug
        if csvpath:
            #subset = self.parse_csv(csvpath)
            self.X_names = list(pd.read_csv(csvpath).file)
            #self.Y_names = [i for i in self.Y_names if i in subset]
        
        if shuffle == True:
            X_names = list(self.X_names)
            random.shuffle(X_names)
            #self.X_names= zip(*X_names)
        
        #assert(len(self.X_names) == len(self.Y_names))
        self.length = len(self.X_names)
        
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[1])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
        
        X = []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]
            #Y_path = self.Y_dir + self.Y_names[j]
            
            #label = [self.PHONEMES.index(yy) for yy in np.load(Y_path)][1:-1]

            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
            #Y.append(np.array(label))
            
        X = np.concatenate(X)#, np.concatenate(Y)
        print(X)
        return X
    
class LibriItems_eval(torch.utils.data.Dataset):
    def __init__(self, X,context = 0):
        #assert(X.shape[0] == Y.shape[0])
        
        self.length  = X.shape[0]
        self.context = context

        if context == 0:
            self.X = X
        else:
            X = np.pad(X, ((context,context), (0,0)), 'constant', constant_values=(0, 0))
            self.X= X
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
            # yy = self.Y[i]
        else:
            xx = self.X[i:(i + 2*self.context + 1)].flatten()
        #print(xx.shape)
        return xx #, yy

    



# %%


def train(args, model, device, train_samples, optimizer, criterion, epoch):
    model.train()
    for i in range(len(train_samples)):
        X, Y = train_samples[i]
        train_items = LibriItems(X, Y, context=args['context'])
        train_loader = torch.utils.data.DataLoader(train_items, batch_size=args['batch_size'], shuffle=True, pin_memory= True, num_workers=4)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, dev_samples):
    model.eval()
    true_y_list = []
    pred_y_list = []
    with torch.no_grad():
        for i in range(len(dev_samples)):
            X, Y = dev_samples[i]

            test_items = LibriItems(X, Y, context=args['context'])
            print(test_items.length)
            test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False, pin_memory= True, num_workers=4)

            for data, true_y in test_loader:
                data = data.float().to(device)
                true_y = true_y.long().to(device)                
                
                output = model(data)
                pred_y = torch.argmax(output, axis=1)
                

                pred_y_list.extend(pred_y.tolist())
                true_y_list.extend(true_y.tolist())

                

    train_accuracy =  accuracy_score(true_y_list, pred_y_list)
    # for x in range(10):
    #     print(pred_y_list[x])
    return train_accuracy, pred_y_list

def eval(args, model, device, test_samples):
    model.eval()
    true_y_list = []
    pred_y_list = []
    with torch.no_grad():
        for i in range(len(test_samples)):
            X= test_samples[i]

            test_items = LibriItems_eval(X, context=args['context'])
            print(test_items.length)
            test_loader = torch.utils.data.DataLoader(test_items, batch_size=args['batch_size'], shuffle=False, pin_memory= True, num_workers=4)

            for data in test_loader:
                data = data.float().to(device)
                #true_y = true_y.long().to(device)                
                
                output = model(data)
                pred_y = torch.argmax(output, axis=1)
                

                pred_y_list.extend(pred_y.tolist())
                #true_y_list.extend(true_y.tolist())

                
    return pred_y_list
    #train_accuracy =  accuracy_score(true_y_list, pred_y_list)
    # for x in range(10):
    #     print(pred_y_list


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network()
    model= nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    # If you want to use full Dataset, please pass None to csvpath
    train_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="train-clean-100", csvpath=None)
    dev_samples = LibriSamples(data_path = args['LIBRI_PATH'], shuffle=True, partition="dev-clean" , csvpath=None)
    
    test_samples = LibriSamples_eval(data_path = args['LIBRI_PATH'], shuffle=True, partition="test-clean" )


        
    for epoch in range(1, args['epoch'] + 1):
        train(args, model, device, train_samples, optimizer, criterion, epoch)
        test_acc, prediction = test(args, model, device, dev_samples)
        prediction_2 = eval(args, model, device, test_samples)

        print('Dev accuracy ', test_acc)
    print(len(prediction_2))
    with open("submission.csv", 'w') as fh:
        fh.write('id,label\n')     
        for x in range(len(prediction_2)):
            fh.write(str(x)+ ',' + str(prediction_2[x]) + "\n")
    #torch.save(model, 'model_6.pt')

if __name__ == '__main__':
    args = {
        'batch_size': 2048,
        'context': 50,
        'log_interval': 200,
        'LIBRI_PATH': '',
        'lr': 0.001,
        'epoch': 50
    }
    main(args)

# %%


# # %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device
# if torch.cuda.device_count() > 0:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
# else:
#   print('no')


# %%



