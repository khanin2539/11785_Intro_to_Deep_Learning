import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extSymbols = []
        skipConnect = []

        for idx, symbol in enumerate(target):
            extSymbols.append(self.BLANK)
            skipConnect.append(False)

            extSymbols.append(symbol)
            condition = idx > 0 and target[idx] != target[idx - 1]
            print(condition)
            skipConnect.append(condition)

        extSymbols.append(self.BLANK)
        skipConnect.append(False)
        extSymbols = np.asarray(extSymbols)
        skipConnect = np.asarray(skipConnect)



        return extSymbols, skipConnect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        alpha[0][0] =  logits[0, extended_symbols[0]]
        # TODO: Intialize alpha[0][1]
        alpha[0][1] = logits[0, extended_symbols[1]]
        
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        for t in range(1, T):
            alpha[t, 0] = alpha[t - 1, 0] * logits[t, extended_symbols[0]]
            for sym in range(1, S):
                if skip_connect[sym]:
                    alpha[t, sym] = alpha[t - 1, sym - 1] + alpha[t - 1, sym] + alpha[t - 1, sym - 2]
                else:
                    alpha[t, sym] = alpha[t - 1, sym - 1] + alpha[t - 1, sym]
                alpha[t, sym] *= logits[t, extended_symbols[sym]]

        return alpha
       


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------
        beta[-1, -1] = 1
        beta[-1, -2] = 1

        for t in reversed(range(T - 1)):
            beta[t, -1] = beta[t + 1, -1] * logits[t + 1, extended_symbols[-1]]
            for r in reversed(range(S - 1)):
                if r + 2 < S - 1 and skip_connect[r + 2]:
                    for i in range(3):
                        beta[t, r] += beta[t + 1, r+i] * logits[t + 1, extended_symbols[r+i]] 
                else:
                    for i in range(2):
                        beta[t, r] += beta[t + 1, r+i] * logits[t + 1, extended_symbols[r+i]] 
                    

        return beta

        

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1).reshape((-1, 1))
        return gamma
     


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            ctc = CTC(self.BLANK)
            logits_batch = logits[0:input_lengths[batch_itr], batch_itr]
            self.extended_symbols, skipConnect = ctc.extend_target_with_blank(target[batch_itr, 0:target_lengths[batch_itr]])
            alpha = ctc.get_forward_probs(logits_batch, self.extended_symbols, skipConnect)
            beta = ctc.get_backward_probs(logits_batch, self.extended_symbols, skipConnect)
            gamma = ctc.get_posterior_probs(alpha, beta)
            for r in range(gamma.shape[1]):
                total_loss[batch_itr] -= np.sum(gamma[0:, r] * np.log(logits_batch[:, self.extended_symbols[r]]))

            self.gammas.append(gamma)

        

        total_loss = np.sum(total_loss) / B
        
        
        return total_loss
        
        

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            gamma = self.gammas[batch_itr]
            ctc = CTC(self.BLANK)
            logits_b = self.logits[0:self.input_lengths[batch_itr], batch_itr]
            extSymbols, _ = ctc.extend_target_with_blank(self.target[batch_itr, 0:self.target_lengths[batch_itr]])
            for r in range(gamma.shape[1]):
                dY[0:self.input_lengths[batch_itr], batch_itr, extSymbols[r]] -= gamma[:, r] / logits_b[:,extSymbols[r]]
            # <---------------------------------------------

        return dY

