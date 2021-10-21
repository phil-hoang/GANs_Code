# Conditional GAN
Based on the tutorial: https://keras.io/examples/generative/conditional_gan/


## Install
In a new environment run

```
pip requirements.txt
```


## Train
Train a model with the `train.py`.
After training is complete, the weights are saved to the folder `weights/exp` where `exp` is appended by a running number.


## Run
Run a previoulsy trained model. For example, to run model `exp2` call.

```
run.py --model exp2
```

Results are then written into the folder `weights/exp2`.


## TODOS
* Add argparse
* In model, remove constants and get parameters from the data.
* Maybe: Split up the model.py, especially the implementation of train_step
* Add automatic saving of weights to folders
* Add logging (single file?)
* Implement run.py
* Add CIFAR10
* Add interface for other datasets
