insturctions to run the code:

1. change the cuda:x to cuda:0 if you have only one GPU
2. run the command  "python hw1p2_script_w_test.py"
3. the submission.csv should be present whern the script is finished. 

Model's architecture:

3 4096 hidden layers 
3 2048 hidden layers 
3 1024 hidden layers 
1 512 hidden layers 
1 256 hidden layers 
1 output layer

Hyperparameters:
Context of 30
epoch: 40
batch size: 2048
learning rate: 0.001


Each hidden layer comes with a dropout of 0.25 and relu activation.


Data Loading scheme:
I followed the smae scheme as the starter notebook by creating a dataloader class for both evaluating and testing, only that testing does not have a Y component.


