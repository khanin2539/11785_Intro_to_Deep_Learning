insturctions to run the code:

1. change the cuda:x to cuda:0 if you have only one GPU
2. run the command  "python hw2p2_try_2_1.py "
3. the submission.csv should be present whern the script is finished. 

Resent Model's architecture:

1 Basic Block:  
    #2 2-D convolutions 
        (3 in-channels-> hidden dimension) kernel size =3 stride =1 padding=1
        batch norm 2D
        Relu
        ( hidden dimension ->  hidden dimension ) kernel size =3 stride =1 padding=1
        batch norm 2D
    if shortcut
        add identity 
    else
        add 1 2-D convolution with ( 3 input-channel -> hidden dimension ) kernel size =1 stride=1 to the network

1 Resnet block
    #1 stem layer containing
        #1 conv2d 
           (3 in-channels->64 hidden dimension) kernel size =3 stride =1 padding=1
            batch norm 2D
            Relu
            Maxpool2d
    #1 cfg layer containing 4 basic blocks
        1). expansion = 1, 3 input channels 64 output channels 3 blocks stride =1
        2). expansion = 1, 3 input channels 128 output channels 4 blocks stride =2
        3). expansion = 1, 3 input channels 256 output channels 6 blocks stride =2
        4). expansion = 1, 3 input channels 512 output channels 4 blocks stride =2
    #1 Adaptive average pooling
    #1 Flatten layer
    #1 linear layer with 512 neurons and 7000 as the output layer


Hyperparameters:

epoch: 67
batch size: 256
learning rate: 0.45 with lr_scheduler.CosineAnnealingLR
optimizer: SGD
criterion: cross entropy
Mixed precision Training
used GPU distributed training

Image transoformation:

ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2)
RandomHorizontalFlip(p=0.2)
RandomRotation(degrees=(-30, 30))
RandomPerspective(distortion_scale=0.2, p=0.2)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

Verification:
used torch.nn.CosineSimilarity(dim=0)


Feature Loading scheme:
I followed the smae scheme as the starter notebook by creating a dataloader class for both evaluating and testing,and added dictaonary to store the features layer from the model.


