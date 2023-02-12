import sys
import math
import numpy as np
from perceptron import perceptron
from confus import confus
from linmach import linmach

# experiment.py

# data = np.loadtxt(sys.argv[1])
# alphas = np.fromstring(sys.argv[2], sep=' ')
# bs = np.fromstring(sys.argv[3], sep=' ')

data = np.loadtxt('/datos/gender.gz')

# alphas = np.fromstring('.1 1 10 100 1000 10000', sep=' ')
alphas = np.fromstring('.1', sep=' ')
# bs = np.fromstring('.1 1 10 100 1000 10000 100000', sep=' ')
bs = np.fromstring('.1 1 10 100 1000 10000 100000', sep=' ')

N,L = data.shape; D=L-1
labs = np.unique(data[:,L-1]); C = labs.size
np.random.seed(23); perm = np.random.permutation(N)
data = data[perm]; NTr = int(round(.7*N))
train = data[:NTr,:]; M = N-NTr; test = data[NTr:,:]

print('#  a  b  E   k  Ete');
print('#------- --- --- ---')
for a in alphas:
  for b in bs:
    w,E,k=perceptron(train,b,a)
    rl=np.zeros((M,1))
    for n in range(M):
      rl[n] = labs[linmach(w,np.concatenate(([1], test[n,:D])))]
    nerr,m = confus(test[:,L-1].reshape(M,1), rl)
    print('%8.1f %8.1f %3d %3d %3d, %3f' % (a, b,E,k,nerr, nerr/M))