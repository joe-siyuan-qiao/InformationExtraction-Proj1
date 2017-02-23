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
                [0 for i in range(self.state_num)]
                for j in range(self.state_num)]
            for state_idx in range(self.state_num):
                self.trellis[-1].node.append(__object__())
                self.trellis[-1].node[-1].alpha = 0.0
                self.trellis[-1].node[-1].beta = 0.0

    def train(self):
        '''
        Reestimate the parameter
        '''
        self.forward()
        self.backward()
        self.calcpost()

    def forward(self):
        # initial state
        for i in range(self.state_num):
            self.trellis[0].node[i].alpha = 1.0 / self.state_num
        # forwarding
        for data_idx in range(1, self.data_len + 1):
            for state_idx in range(self.state_num):
                acc_alpha = 0.0
                for old_idx in range(self.state_num):
                    acc_alpha = acc_alpha + self.trans_mat[old_idx][state_idx]
                        * self.emiss_mat[old_idx][self.data[data_idx] - 1]
                        * self.trellis[data_idx - 1].node[old_idx].alpha
                self.trellis[data_idx].node[state_idx].alpha = acc_alpha
            acc_alpha = 0.0
            for state_idx in range(self.state_num):
                acc_alpha += self.trellis[data_idx].node[state_idx].alpha
            self.trellis[data_idx].norm = acc_alpha
            for state_idx in range(self.state_num):
                self.trellis[data_idx].node[state_idx].alpha /= acc_alpha

    def backward(self):
        # initial state
        for i in range(self.state_num):
            self.trellis[-1].node[i].beta = 1.0 / self.trellis[-1].norm
        for data_idx in range(self.data_len - 1, -1, -1):
            for state_idx in range(self.state_num):
                acc_beta = 0.0
                for old_idx in range(self.state_num):
                    acc_beta = acc_beta + self.trans_mat[state_idx][old_idx]
                        * self.emiss_mat[old_idx][self.data[data_idx + 1] - 1]
                        * self.trellis[data_idx + 1].node[old_idx].beta
                acc_beta /= self.trellis[data_idx].norm
                self.trellis[data_idx].node[state_idx].beta = acc_beta

    def calcpost(self):
        for data_idx in range(self.data_len):
            for left_idx in range(self.state_num):
                for right_idx in range(self.state_num):
                    alpha = self.trellis[data_idx].node[left_idx].alpha
                    trans = self.trans_mat[left_idx][right_idx]
                    emiss = self.emiss_mat[right_idx][self.data[data_idx + 1]]
                    beta = self.trellis[data_idx + 1].node[right_idx].beta
                    prod = alpha * trans * emiss * beta
                    self.trellis[data_idx].post[left_idx][right_idx] = prod


class __object__:
    pass
