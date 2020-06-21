import numpy as np

tensor_prefix = "center"

tensor_old = np.load("{}_old.npz".format(tensor_prefix), allow_pickle=True)
tensor_new = np.load("{}_new.npz".format(tensor_prefix), allow_pickle=True)

err = np.abs(tensor_old - tensor_new).max()
print("{} max_err: {}".format(tensor_prefix, err))

from IPython import embed
embed()
