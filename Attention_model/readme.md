insturctions to run the code:

2. run the command  "python student_hw4p2_starter_notebook_1.py"
3. the hw4p2_submission.csv should be present whern the script is finished. 

Resent Model's architecture:

Seq2Seq(
  (encoder): Encoder(
    (lstm): LSTM(13, 256, batch_first=True, bidirectional=True)
    (pBLSTMs): Sequential(
      (0): pBLSTM(
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
      (1): LockedDropout(p=0.2)
      (2): pBLSTM(
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
      (3): LockedDropout(p=0.2)
      (4): pBLSTM(
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
      (5): LockedDropout(p=0.2)
    )
    (key_network): Linear(in_features=512, out_features=128, bias=True)
    (value_network): Linear(in_features=512, out_features=128, bias=True)
  )
  (decoder): Decoder(
    (embedding): Embedding(30, 256, padding_idx=29)
    (lstm1): LSTMCell(384, 256)
    (lstm2): LSTMCell(256, 128)
    (attention): Attention()
    (character_prob): Linear(in_features=256, out_features=30, bias=True)
  )
)

Hyperparameters:

epoch: 100
batch size: 64/128
learning rate: 0.002 with lr_scheduler
teacher forcing: starting from 0.7 and reduce -0.05 for each 5 epochs starting from epoch >20
optimizer: Adam
criterion: CrossEntropyLoss

Feature Loading scheme:
I followed the smae scheme as the starter notebook of HW3P2 by creating a dataloader class for training and testing seperately, each of which contains a collate function.


