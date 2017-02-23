"""
The implementation of the hidden markov model
"""


class model:

    def __init__(self, init_trans_mat, init_emiss_mat):
        '''
        Initialize the model

        Parameters
        ----------
        init_trans_mat: [] of []
            initial transition matrix
        init_emiss_mat: [] of []
            initial emission matrix
        '''
        self.trans_mat = init_trans_mat
        self.emiss_mat = init_emiss_mat
        self.state_num = len(self.trans_mat)

    def read(self, filename):
        '''
        Read the text file and save to self.data

        Parameters
        ----------
        filename: string
            the path of the text file
        '''
        self.data = []
        fin = open(filename, 'r')
        line = fin.read()
        for letter in line:
            if letter == ' ':
                self.data.append(27)
            else:
                self.data.append(ord(letter) - ord('a') + 1)
        self.data_len = len(self.data)
        # build the model
        self.trellis = []
        for data_idx in range(self.data_len + 1):
            self.trellis.append(__object__())
            self.trellis[-1].node = []
            self.trellis[-1].norm = 1.0
            self.trellis[-1].post = [
                [0 for i in range(self.state_num)] for j in self.state_num]
            for state_idx in range(self.state_num):
                self.trellis[-1].node.append(__object__())
                self.trellis[-1].node[-1].alpha = 0.0
                self.trellis[-1].node[-1].beta = 0.0


class __object__:
    pass
