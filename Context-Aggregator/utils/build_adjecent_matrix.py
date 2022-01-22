import itertools
import numpy as np
import torch
from utils.torchutil import to_sparse
import pdb
def build_adjecent_matrix(n):
    adjecent_matrix = np.zeros((72,72), dtype=np.float32)
    vertices = list(itertools.permutations(range(n), 2))
    for i, x in enumerate(vertices):
        row_sum = 0.0
        for j, y in enumerate(vertices):
            adjecent_matrix[i][j] = 1 if x[0] == y[1] or x[1] == y[0] or i == j else 0 
            row_sum += adjecent_matrix[i][j]
        if(row_sum != 0.0):
            adjecent_matrix[i] /= row_sum    
    AM = torch.from_numpy(adjecent_matrix)
    return AM

adjecent_matrix = []
for i in range(2, 10):
    AM = build_adjecent_matrix(i)
    adjecent_matrix.append(AM)
    # adjecent_matrix.append(torch.eye(72))

def main():
    for i in range(8):
        print(adjecent_matrix[i])

if __name__ == "__main__":
    main()
