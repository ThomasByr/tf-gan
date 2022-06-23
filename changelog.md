# Changelog

<summary>The full history, or so was I told...</summary>

## Beta first minor release

**v0.1.0** first release

- initial commit
- explicitly import estimators from tensorflow as a separate import instance
- removed extensive backward python compatibility
- removed refere,ces tp g-no-augmented assignment in pylint directives

**v0.1.1** the first patch

- update tf imports so examples work again
- no-op
- add new paper
- don't fail if the desired python version is already installed
- remove testing for 1.x
- update mnist example
- update gan tpu
- fix images in cyclegan
- upgraded deprecated `inspect.getargspec()`
- data provider for dme_cyclegan and implement logic to allow specification of files with a list of image paths to be used for training for dme_cyclegan.
- update on tpu : use `tf.compat.v1` when necessary
- fix order dependent tests

**v0.1.2** this is getting serious, I guess

- add regan loss function
- update `losses_impl_test.py`
- update on generic parameters
- lazy module loading
- fix horrible variable names in `tpu_gan_estimator_test`
- do not propagate parent name_scope in v2 control flow when inside of v1
- specify tensorflow dataset version to use
- tf2 rename and reenable tests
- fix bugs where examples fail with "doesn't work when executing eagerly
- update release to use `python3.7` as 3.6 venv seem busted on Debian

## Stable release 1

**v1.0.0** big day

- am I going to ever get paid for this ?
- 2d batch normalization with gamma and beta was broadcasted incorrectly
- minor additions to the eval helper functions
- add tfhub to dependencies
- pin tensorflow-probability to version 0.8 for tf1 since that's the last version that supports tf1
- switch from matplotlib to pillow
- add dummy computation to trigger method lazy loading before mocking happens
- make tf-gan default build `python3.8`
- remove testing support for python 2.x

**v1.1.0** breakthrough do not happend with legacy codes

- removed partial support for `python3.8` and lower (might pin python version to `3.10` in the future)
- removed service for python `<=3.6`
- updated `lib_eval.py`, `data_provider.py`
- added relativistic loss and loss test
- reworked `networks.py`, `train_lib.py`, `utils.py`
