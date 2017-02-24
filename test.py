import sys, os
sys.path.insert(0, 'python')
import hmm

with open(os.path.join('data', 'textA.txt')) as f:
    train_data = f.read()

with open(os.path.join('data', 'textB.txt')) as f:
    test_data = f.read()

init_trans_mat = []
init_emiss_mat = []
fin = open(os.path.join('config', 'init_trans_mat_2'))
while True:
    line = fin.readline()
    if line == '':
        break
    if line[-1] == '\n':
        line = line[:-1]
    line = line.split()
    init_trans_mat.append([])
    for number in line:
        init_trans_mat[-1].append(float(number))
fin.close()
fin = open(os.path.join('config', 'init_emiss_mat_2'))
while True:
    line = fin.readline()
    if line == '':
        break
    if line[-1] == '\n':
        line = line[:-1]
    line = line.split()
    init_emiss_mat.append([])
    for number in line:
        init_emiss_mat[-1].append(float(number))

model = hmm.model(init_trans_mat, init_emiss_mat)
model.read('data/textA.txt')
model.train()
