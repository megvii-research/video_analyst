# SOT TEST 

A collection of test scripts are located under _tools/test/_:

- [tools/test/test_VOT.sh](../../tools/test/test_VOT.sh)
- [tools/test/test_GOT.sh](../../tools/test/test_GOT.sh)

## Check test results

_EXP_NAME_ is the string value of key _test.track.exp_name_ in the corresponding _.yaml_ file.

### Check VOT results

```Bash
view logs/VOT2018/*.csv
```

### Check GOT results

```Bash
view logs/GOT-Benchmark/report/GOT-10k/EXP_NAME/performance.json
```
