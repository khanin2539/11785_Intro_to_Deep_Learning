import numpy as np

from typing import Dict, List
def clean_path(path):
	""" utility function that performs basic text cleaning on path """

	# No need to modify
	path = str(path).replace("'","")
	path = path.replace(",","")
	path = path.replace(" ","")
	path = path.replace("[","")
	path = path.replace("]","")

	return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        SymbolSets = self.symbol_set

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path

        print(SymbolSets)
        print(y_probs)
        # p = 1
        for i in range(y_probs.shape[1]):
            # print(y_probs[t])
            # print(np.max(y_probs[:, t, 0]))
            path_prob*= np.max(y_probs[:, i, 0])
            # print(p)
            idx = np.argmax(y_probs[:, i, 0])

            # print(index)
            if idx != 0:
                if not blank:
                    if len(decoded_path) == 0 or decoded_path[-1] != self.symbol_set[idx - 1]:
                        decoded_path.append(self.symbol_set[idx - 1])
                    
                else:
                    if len(decoded_path) == 0 or decoded_path[-1] != self.symbol_set[idx - 1]:
                        # print(index)
                        
                        decoded_path.append(self.symbol_set[idx -1])
                        blank = 1
            else:
                
                blank = 1

        decoded_path = clean_path(decoded_path)

        return  decoded_path, path_prob

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """
        self.symbol_set = symbol_set
        self.beam_width = beam_width
        
    def extend_with_symbol(self, t, y_probs):
        updated_paths_terminal_blank = []
        updated_paths_terminal_blank_score = {}

        for path in self.paths_blank:
            for idx, c in enumerate(self.symbol_set):
                new_path = path + c
                updated_paths_terminal_blank.append(new_path)
                updated_paths_terminal_blank_score[new_path] = self.paths_blank_score[path] * y_probs[
                    idx + 1, t, 0]

        for path in self.paths_symbol:
            for idx, c in enumerate(self.symbol_set):
                new_path = path if c == path[-1] else path + c
                if new_path in updated_paths_terminal_blank_score:
                    updated_paths_terminal_blank_score[new_path] += self.paths_symbol_score[path] * y_probs[idx + 1, t, 0]
                else:
                    updated_paths_terminal_blank_score[new_path] = self.paths_symbol_score[path] *  y_probs[idx + 1, t, 0]
                    updated_paths_terminal_blank.append(new_path)

        return updated_paths_terminal_blank, updated_paths_terminal_blank_score

    def extend_with_blank(self, t, y_probs):
        updated_paths_terminal_blank = []
        updated_paths_terminal_blank_score = {}

        for path in self.paths_blank:
            updated_paths_terminal_blank.append(path)
            updated_paths_terminal_blank_score[path] = self.paths_blank_score[path] * y_probs[0, t, 0]

        for path in self.paths_symbol:
            if path in updated_paths_terminal_blank:
                updated_paths_terminal_blank_score[path] += self.paths_symbol_score[path] * y_probs[
                    0, t, 0]
            else:
                updated_paths_terminal_blank_score[path] = self.paths_symbol_score[path] * y_probs[
                    0, t, 0]
                updated_paths_terminal_blank.append(path)

        return updated_paths_terminal_blank, updated_paths_terminal_blank_score

    def prune(self):
        prune_updated_paths_blank = []
        prune_updated_paths_blank_score = {}

        prune_updated_paths_symbol = []
        prune_updated_paths_symbol_score = {}

        scorelist = []
        # First gather all the relevant scores
        for score in self.paths_blank_score.values():
            scorelist.append(score)

        for score in self.paths_symbol_score.values():
            scorelist.append(score)
        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist.sort()

        if len(scorelist) < self.beam_width:
            cutoff = scorelist[-1] 
        else:
            cutoff =  scorelist[- self.beam_width]

        for path in self.paths_blank:
            if self.paths_blank_score[path] >= cutoff:
                prune_updated_paths_blank.append(path)  # Set addition
                prune_updated_paths_blank_score[path] = self.paths_blank_score[path]

        for path in self.paths_symbol:
            if self.paths_symbol_score[path] >= cutoff:
                prune_updated_paths_symbol.append(path) # Set addition
                prune_updated_paths_symbol_score[path] = self.paths_symbol_score[path]

        self.paths_blank = prune_updated_paths_blank
        self.paths_blank_score = prune_updated_paths_blank_score
        self.paths_symbol = prune_updated_paths_symbol
        self.paths_symbol_score = prune_updated_paths_symbol_score
        

    def merge(self):
        MergedPaths = self.paths_blank
        FinalPathScore = self.paths_blank_score

        for path in self.paths_symbol:
            if path in MergedPaths:
                FinalPathScore[path] += self.paths_symbol_score[path]
            else:
                MergedPaths.append(path)  # Set addition
                FinalPathScore[path] = self.paths_symbol_score[path]

        max_path = MergedPaths[0]
        max_score = FinalPathScore[max_path]
        for path in FinalPathScore:
            if FinalPathScore[path] > max_score:
                max_path = path
                max_score = FinalPathScore[path]

        return max_path, FinalPathScore
    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        
        decoded_path = []
        sequences = [[list(), 1.0]]
        ordered = None

        best_path, merged_path_scores = None, None

        
        self.paths_blank: List[str] = ['']
        self.paths_blank_score: Dict[str:np.ndarray] = {'': y_probs[0, 0, 0]}

        self.paths_symbol: List[str] = [c for c in self.symbol_set]
        self.paths_symbol_score: Dict[str:np.ndarray] = {}
        for i, c in enumerate(self.symbol_set):
            self.paths_symbol_score[c] = y_probs[i + 1, 0, 0]
        for t in range(1, y_probs.shape[1]):
            self.prune()
            updated_paths_symbol, updated_paths_symbol_score = self.extend_with_symbol(t, y_probs)
            updated_paths_blank, updated_paths_blank_score = self.extend_with_blank(t, y_probs)
            self.paths_blank = updated_paths_blank
            self.paths_symbol = updated_paths_symbol
            self.paths_blank_score = updated_paths_blank_score
            self.paths_symbol_score = updated_paths_symbol_score

        return self.merge()
        
    
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return      

        # return best_path, merged_path_scores
        raise NotImplementedError
