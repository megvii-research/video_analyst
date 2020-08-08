import numpy as np
# from IPython import embed;embed()
from make_densebox_target import \
    make_densebox_target as make_densebox_target_old
from make_densebox_target_dev import \
    make_densebox_target as make_densebox_target_new

gt_boxes = np.asarray([[150, 250, 130, 60, 1]])
config_dict = dict(
    x_size=303,
    score_size=17,
    total_stride=8,
    score_offset=(303 - 1 - (17 - 1) * 8) // 2,
)

target_old = make_densebox_target_old(gt_boxes, config_dict)
target_new = make_densebox_target_new(gt_boxes, config_dict)

for v_old, v_new in zip(target_old, target_new):
    v_new = v_new.numpy()
    # uncomment the next line to inspect tensors in detail
    # from IPython import embed;embed()
    np.testing.assert_allclose(v_new, v_old, atol=1e-6, verbose=True)
    print("Values closed.")
