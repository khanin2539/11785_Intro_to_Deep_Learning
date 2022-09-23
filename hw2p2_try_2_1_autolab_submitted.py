# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as ttf
import math
import os
import os.path as osp

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

torch.cuda.empty_cache()
# %%


# %% [markdown]
# # TODOs
# As you go, please read the code and keep an eye out for TODOs!

#
"""
The well-accepted SGD batch_size & lr combination for CNN classification is 256 batch size for 0.1 learning rate.
When changing batch size for SGD, follow the linear scaling rule - halving batch size -> halve learning rate, etc.
This is less theoretically supported for Adam, but in my experience, it's a decent ballpark estimate.
"""



batch_size = 256
lr = 0.45
epochs = 67 # Just for the early submission. We'd want you to train like 50 epochs for your main submissions.
print(batch_size)
# %% [markdown]
# # Very Simple Network

# %%

class BasicBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride, expansion
                ):
        super().__init__() # Just have to do this for all nn.Module classes
        expansion = 1
        hidden_dim = out_channels*expansion
        # Can only do identity residual connection if input & output are the
        # same channel & spatial shape.
        if stride == 1 and in_channels == out_channels:
            self.shortcut = True
        else:
            self.shortcut = False

        self.block = nn.Sequential(
            # TODO: Fill this in!
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=stride, padding=1, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(out_channels, hidden_dim, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.relu = nn.ReLU()
        
        # if self.shortcut == False:
        self.do_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, stride, bias = False),
            nn.BatchNorm2d(hidden_dim),
        )
        
       

    def forward(self, x):
 

        out = self.block(x)

        if self.shortcut:
            x = x + out
        else:
            x = (self.do_conv(x) + out)
        return x



class ResNet(nn.Module):

    def __init__(self, num_classes= 7000, dropout=0.3):
        super().__init__()

        self.num_classes = num_classes
        self.in_planes = 64

        self.stem = nn.Sequential(
            # TODO: Fill this in!
            nn.Conv2d(3, self.in_planes, kernel_size=7, stride = 2, padding = 3,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding = 1)
            
        )

        # Since we're just repeating InvertedResidualBlocks again and again, we
        # want to specify their parameters like this.
        # The four numbers in each row (a stage) are shown below.
        # - Expand ratio: We talked about this in InvertedResidualBlock
        # - Channels: This specifies the channel size before expansion
        # - # blocks: Each stage has many blocks, how many?
        # - Stride of first block: For some stages, we want to downsample. In a
        #   downsampling stage, we set the first block in that stage to have
        #   stride = 2, and the rest just have stride = 1.

        # Again, note that almost every stage here is downsampling! By the time
        # we get to the last stage, what is the image resolution? Can it still
        # be called an image for our dataset? Think about this, and make changes
        # as you want. 
        
        self.stage_cfgs = [
            # expand_ratio, channels, # blocks, stride of first block
            [1, 64, 3, 1],
            [1, 128, 4, 2],
            [1, 256, 6, 2],
            [1, 512, 3, 2],
        ]

        # Remember that our stem left us off at 16 channels. We're going to 
        # keep updating this in_channels variable as we go
        in_channels = 64

        # Let's make the layers
        layers = []
        for curr_stage in self.stage_cfgs:
            expansion, num_channels, num_blocks, stride = curr_stage
            
            for block_idx in range(num_blocks):
                out_channels = num_channels
                layers.append(BasicBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        # only have non-trivial stride if first block
                        stride=stride if block_idx == 0 else 1, 
                        expansion = expansion
                ))
                # In channels of the next block is the out_channels of the current one
                in_channels = out_channels 
            
        self.layers = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()) # Done, save them to the class
        
        # Now, we need to build the final classification layer.
        self.cls_layer = nn.Sequential(
            
            nn.Dropout(p=dropout),
            nn.Linear(512, self.num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x,  return_feats=False):
        out = self.stem(x)
        feats = self.layers(out)


        if return_feats==True:
            return feats
        else:
            out = self.cls_layer(feats)
            return out

"""
Transforms (data augmentation) is quite important for this task.
Go explore https://pytorch.org/vision/stable/transforms.html for more details
"""
DATA_DIR = ""
TRAIN_DIR = osp.join(DATA_DIR, "classification/classification/train") # This is a smaller subset of the data. Should change this to classification/classification/train
VAL_DIR = osp.join(DATA_DIR, "classification/classification/dev")
TEST_DIR = osp.join(DATA_DIR, "classification/classification/test")

# train_transforms = [ttf.ToTensor()]
transform = torchvision.transforms.Compose([
                #torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
                torchvision.transforms.RandomHorizontalFlip(p=0.4),
                torchvision.transforms.RandomRotation(degrees=(-30, 30)),
                torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

val_transforms = [ttf.ToTensor()]

train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                 transform=transform)
val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                               transform=transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, drop_last=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, num_workers=8)
model = ResNet()
# #model = model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= nn.DataParallel(model)
# model= nn.DataParallel(model, device_ids = [2, 3])
# model.to(f'cuda:{model.device_ids[0]}')
model.to(device)

# For this homework, we're limiting you to 35 million trainable parameters, as
# outputted by this. This is to help constrain your search space and maintain
# reasonable training times & expectations
num_trainable_parameters = 0
for p in model.parameters():
    num_trainable_parameters += p.numel()
print("Number of Params: {}".format(num_trainable_parameters))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))
# T_max is "how many times will i call scheduler.step() until it reaches 0 lr?"

# For this homework, we strongly strongly recommend using FP16 to speed up training.
# It helps more for larger models.
# Go to https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision
# and compare "Single precision training" section with "Mixed precision training" section
scaler = torch.cuda.amp.GradScaler()

# # %% [markdown]
# # # Dataset & DataLoader

# # %%




for epoch in range(epochs):
    # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(torch.device('cuda:0'))
        y = y.to(torch.device('cuda:0'))

        # Don't be surprised - we just wrap these two lines to make it work for FP16
        with torch.cuda.amp.autocast():     
            outputs = model(x)
            loss = criterion(outputs, y)

        # Update # correct & loss as we go
        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total_loss += float(loss)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        
        # Another couple things you need for FP16. 
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

        batch_bar.update() # Update tqdm bar
    batch_bar.close() # You need this to close the tqdm bar
    print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
        epoch + 1,
        epochs,
        100 * num_correct / (len(train_loader) * batch_size),
        float(total_loss / len(train_loader)),
        float(optimizer.param_groups[0]['lr'])))
    # model_name = 'resnet_34_flatten_hw2p2_test_early_submission.pt'
    # torch.save(model, model_name)


model = torch.load('resnet_34_flatten_hw2p2_test_early_submission.pt')


# %%
model.eval()
batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
num_correct = 0
for i, (x, y) in enumerate(val_loader):

    x = x.to(torch.device('cuda:0'))
    y = y.to(torch.device('cuda:0'))

    with torch.no_grad():
        outputs = model(x)
    # hey_to_list = torch.argmax(outputs, axis=1).tolist()
    #print(hey_to_list)
    num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
    batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)))

    batch_bar.update()
    
batch_bar.close()
print("Validation: {:.04f}%".format(100 * num_correct / len(val_dataset)))

# %% [markdown]
# # Classification Task: Submit to Kaggle

# %%
class ClassificationTestSet(Dataset):
    # It's possible to load test set data using ImageFolder without making a custom class.
    # See if you can think it through!

    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in data_dir
        self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))

# %%
test_transforms = ttf.Compose([
                    ttf.ToTensor(),
                    ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

transform = torchvision.transforms.Compose([
                #torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
                torchvision.transforms.RandomHorizontalFlip(p=0.2),
                torchvision.transforms.RandomRotation(degrees=(-30, 30)),
                torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
test_dataset = ClassificationTestSet(TEST_DIR, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=8)

# %%
model.eval()
batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')

res = []
for i, (x) in enumerate(test_loader):

    # TODO: Finish predicting on the test set.
    x = x.to(torch.device('cuda:0'))
    

    with torch.no_grad():
        outputs = model(x)
        hey_to_list = torch.argmax(outputs, axis=1).tolist()
    res.extend(hey_to_list)
    #print(len(res))

    

    batch_bar.update()
print(len(res))
batch_bar.close()

# %%
with open("resnet_flatten_test_classification_early_submission_1.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(test_dataset)):
        f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", res[i]))


class VerificationDataset(Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in data_dir
        self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # We return the image, as well as the path to that image (relative path)
        return self.transforms(Image.open(self.img_paths[idx])), osp.relpath(self.img_paths[idx], self.data_dir)

# %%
test_transforms = ttf.Compose([
                    ttf.ToTensor(),
                    ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
test_dataset = ClassificationTestSet(TEST_DIR,test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=8)

# %%
model.eval()
val_veri_dataset = VerificationDataset(osp.join(DATA_DIR, "verification/verification/dev"),
                                       ttf.Compose(val_transforms))
val_ver_loader = torch.utils.data.DataLoader(val_veri_dataset, batch_size=batch_size, 
                                             shuffle=False, num_workers=8)
feats_dict = dict()

for batch_idx, (imgs, path_names) in tqdm(enumerate(val_ver_loader), total=len(val_ver_loader), position=0, leave=False):
    imgs = imgs.to(torch.device('cuda:0'))

    with torch.no_grad():
        # Note that we return the feats here, not the final outputs
        # Feel free to try the final outputs too!
        # feats = model(imgs, return_feats=True) 
        # f = nn.GELU()
         
        
        feats = model(imgs, return_feats=True) 
        # feats = f(feats)
 
    feats_dict.update({**feats_dict, **dict(zip(path_names, feats))}) 
    # TODO: Now we have features and the image path names. What to do with them?
    # Hint: use the feats_dict somehow.

# %%
# What does this dict look like?
print(list(feats_dict.items())[0])

# %%
# We use cosine similarity between feature embeddings.
# TODO: Find the relevant function in pytorch and read its documentation.
similarity_metric = torch.nn.CosineSimilarity(dim=0)

val_veri_csv = osp.join(DATA_DIR, "verification/verification/verification_dev.csv")


# Now, loop through the csv and compare each pair, getting the similarity between them
pred_similarities = []
gt_similarities = []
for line in tqdm(open(val_veri_csv).read().splitlines()[1:], position=0, leave=False): # skip header
    img_path1, img_path2, gt = line.split(",")

    # TODO: Use the similarity metric
    # How to use these img_paths? What to do with the features?
    similarity = similarity_metric(feats_dict[img_path1.split('/')[-1]], feats_dict[img_path2.split('/')[-1]])

    pred_similarities.append(similarity.item())

    gt_similarities.append(int(gt))

pred_similarities = np.array(pred_similarities)
gt_similarities = np.array(gt_similarities)

print("AUC:", roc_auc_score(gt_similarities, pred_similarities))

test_veri_dataset = VerificationDataset(osp.join(DATA_DIR, "verification/verification/test"),
                                        ttf.Compose(val_transforms))
test_ver_loader = torch.utils.data.DataLoader(test_veri_dataset, batch_size=batch_size, 
                                              shuffle=False, num_workers=8)

# %%
model.eval()

feats_dict = dict()
for batch_idx, (imgs, path_names) in tqdm(enumerate(test_ver_loader), total=len(test_ver_loader), position=0, leave=False):
    imgs = imgs.to(torch.device('cuda:0'))

    with torch.no_grad():
        # Note that we return the feats here, not the final outputs
        # Feel free to try to final outputs too!
        # f = nn.GELU()
        feats = model(imgs, return_feats=True) 
        # feats = f(feats)
    
    # TODO: Now we have features and the image path names. What to do with them?
    # Hint: use the feats_dict somehow.
    feats_dict.update({**feats_dict, **dict(zip(path_names, feats))}) 

# %%
# We use cosine similarity between feature embeddings.
# TODO: Find the relevant function in pytorch and read its documentation.
similarity_metric = torch.nn.CosineSimilarity(dim=0) 
val_veri_csv = osp.join(DATA_DIR, "verification/verification/verification_test.csv")


# Now, loop through the csv and compare each pair, getting the similarity between them
pred_similarities = []
for line in tqdm(open(val_veri_csv).read().splitlines()[1:], position=0, leave=False): # skip header
    img_path1, img_path2 = line.split(",")

    # TODO: Finish up verification testing.
    # How to use these img_paths? What to do with the features?
    similarity = similarity_metric(feats_dict[img_path1.split('/')[-1]], feats_dict[img_path2.split('/')[-1]])

    pred_similarities.append(similarity.item())

# %%
with open("verification_early_submission_1.csv", "w+") as f:
    f.write("id,match\n")
    for i in range(len(pred_similarities)):
        f.write("{},{}\n".format(i, pred_similarities[i]))


torch.cuda.empty_cache()

