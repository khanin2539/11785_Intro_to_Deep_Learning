import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Understand then uncomment this code :)
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size)
            Input

        h_0: (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits: (batch_size, output_size)

        Output: logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros(
                (self.num_layers, batch_size, self.hidden_size), dtype=float
            )
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = None

        ### Add your code here --->
        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
        for x in range(seq_len):
                        #[batch_size, seq_len, input_size]
            input = self.x[:, x, :]
            hidden = []
        #   Iterate over the length of your self.rnn (through the layers)
            for y in range(len(self.rnn)):
                #       Run the rnn cell with the correct parameters and update
                h_ti = self.rnn[y].forward(input, self.hiddens[-1][y])
            #       the parameters as needed. Update hidden.
                input = h_ti
                
                hidden.append(h_ti)
            #   Similar to above, append a copy of the current hidden array to the hiddens list
            self.hiddens.append(hidden.copy())
     # Get the outputs from the last time step using the linear layer and return it
        # <--------------------------
        # similar to linear(forward)
        logits = self.output_layer.forward(input)
        # TODO

        print(self.hiddens)
        
        

        return logits
        # raise NotImplementedError

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        """

        * Notes:
        * More specific pseudocode may exist in lecture slides and a visualization
          exists in the writeup.
        * WATCH out for off by 1 errors due to implementation decisions.

        Pseudocode:
        * Iterate in reverse order of time (from seq_len-1 to 0)
            * Iterate in reverse order of layers (from num_layers-1 to 0)
                * Get h_prev_l either from hiddens or x depending on the layer
                    (Recall that hiddens has an extra initial hidden state)
                * Use dh and hiddens to get the other parameters for the backward method
                    (Recall that hiddens has an extra initial hidden state)
                * Update dh with the new dh from the backward pass of the rnn cell
                * If you aren't at the first layer, you will want to add
                  dx to the gradient from l-1th layer.

        * Normalize dh by batch_size since initial hidden states are also treated
          as parameters of the network (divide by batch size)

        """
        # TODO
        #loop through time
        for x in reversed(range(seq_len)):
            #loop through layer
            for y in reversed( range(self.num_layers)):
                # Get h_prev_l either from hiddens or x depending on the layer
                #     (Recall that hiddens has an extra initial hidden state)
                if y!=0:

                    h_prev_l = self.hiddens[x+1][y-1]
                else: h_prev_l = self.x[:, x, :]

                h_prev_t = self.hiddens[x][y]
                print(x, y)
                # Use dh and hiddens to get the other parameters for the backward method
                #     (Recall that hiddens has an extra initial hidden state)
                dx_, dh_ = self.rnn[y].backward(dh[y], self.hiddens[x+1][y], h_prev_l, h_prev_t)
                
                # Update dh with the new dh from the backward pass of the rnn cell
                dh[y] =dh_

                if y!=0:
                    dh[y-1]+=dx_
        # dh = dh/batch_size


        return dh / batch_size
        # raise NotImplementedError
