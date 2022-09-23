insturctions to run the code:

2. run the command  "python HW3P2_1_128_1.py"
3. the submission.csv should be present whern the script is finished. 

Resent Model's architecture:

1 Basic Block:  
    #2 1-D convolutions 
        (13 in-channels-> hidden dimension = 256) kernel size =3 stride =2 padding=1
        batch norm 1D
        Relu
        ( hidden dimension = 256->  hidden dimension = 256) kernel size =3 stride =1 padding=1
        batch norm 2D
    if shortcut
        add identity 
    else
        add 1 1-D convolution with ( 13 input-channel -> hidden dimension = 256 ) kernel size =1 stride=1 to the network

1 Network block
    #1 Basic Block
    #1 LSTM block with 256 input size hiddden size of 256 and 4 layers bidiectional True
    
1 Classification Layer
    #1 linear 1024 neurons -> 2048 neurons
    #1 linear 2048 -> 41 output layers
    

Hyperparameters:

epoch: 50
batch size: 64
learning rate: 0.002 with lr_scheduler.CosineAnnealingLR
optimizer: Adam
criterion: CTC loss

Feature Loading scheme:
I followed the smae scheme as the starter notebook of HW1P2 by creating a dataloader class for training and testing seperately, each of which contains a collate function.


