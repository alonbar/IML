import numpy as np

a = np.matrix('50 30; 30 50')
print(np.linalg.eig(a)[1])