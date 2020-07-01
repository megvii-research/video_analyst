import numpy as np

DUMP_DIR = "dump"
tensor_prefix = "center"
tensor_suffix_old = "main"
tensor_suffix_new = "dev"

tensor_old = np.load("{}/{}_{}.npz".format(DUMP_DIR, tensor_prefix,
                                           tensor_suffix_old),
                     allow_pickle=True)
tensor_new = np.load("{}/{}_{}.npz".format(DUMP_DIR, tensor_prefix,
                                           tensor_suffix_new),
                     allow_pickle=True)

# uncomment the next line to inspect tensors in detail
# from IPython import embed;embed()
np.testing.assert_allclose(
    tensor_new,
    tensor_old,
    atol=1e-6,
    err_msg="Value not cloased for tensor: {}".format(tensor_prefix),
    verbose=True)
print("Values cloased for tensor: {}".format(tensor_prefix))
