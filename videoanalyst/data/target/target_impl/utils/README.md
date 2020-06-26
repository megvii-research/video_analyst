# About debugging in this part

This part is resposible for training target generation, one of the most critical part in this codebase. Thus debugging codes have been installed inside and tools are provided for unit testing to ensure the correctness.

## Comparing the target

Assuming comparison between _make_densebox_target.py_ and _make_densebox_target_v1.py_

Set _DUMP_FLAG_ to _True_ in both _make_densebox_target.py_ and _make_densebox_target_v1.py_

Set _gt_boxes_ in _debug_compare_densebox_target.py_ to desired testing inputs.

```
python debug_compare_densebox_target.py
```

See "Values closed" to ensure that targets generated are "acceptably equal" (error within tolerance).

## Comparing the intermediate tensors

Set _tensor_prefix_ in _debug_compare_tensor.py_.

```
python debug_compare_densebox_target.py
python debug_compare_tensor.py
```

## After debugging

Clean dumped tensors.

```
rm -r dump
```
