# Hyper-Parameter Search

Methods for video tasks usually contain a set of hyper-parameters and their combination of value can . Thus Hyper-Parameter Optimization (HPO).

Note that we only encourage HPO on validation subset of dataset. Any HPO on test subset will result in data leak by definition and such results will be "illegal" to publish.

## Usage

```bash
python3 main/hpo.py --config=experiments/siamfcpp/test/vot/siamfcpp_alexnet-multi_temp.yaml --hpo-config=experiments/siamfcpp/hpo/siamfcpp_SiamFCppMultiTempTracker-hpo.yaml
```

* _--config_ can be any experiment test
* _--hpo-config_ is the hpo config.
  * exp_save is specified in this hpo config .yaml file (by default _logs/hpo_).

## About HPO method

Currently, we use naive Random Search algorithm (e.g. [RandomSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)) for its simplicity in implementation.

We are open to recommandation of other efficient and effective HPO methods. 
