"""
@Time    : 2024/3/11 08:18
@Author  : YuZhang
@File    : tools.py
"""
import numpy as np
import torch
import random
def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

# Random arrays
def shuffle(*arrays, **kwargs):
    # indices? Whether return the shuffle_indices(True/False)
    require_indices = kwargs.get('indices', False)

    # Determine whether the lengths are equal. If equal, the length of the set is 1
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('Inputs to shuffles must be have the same length')

    # We set seed before, so we can shuffle the index in the arrays(csr_matrix)
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
def mini_batch(*arrays, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)

    if len(arrays) == 1:
        for i in range(0, len(arrays[0]), batch_size):
            yield arrays[0][i: i+batch_size]
    else:
        for i in range(0, len(arrays[0]), batch_size):
            yield tuple(array[i: i+batch_size] for array in arrays)

def convert_sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    value = torch.FloatTensor(coo.data)
    sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
    sp_tensor = sp_tensor.coalesce()
    return sp_tensor