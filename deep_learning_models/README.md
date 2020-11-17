# Usage

To run jupyter notebooks from AWS instance:

1) ssh with port forwarding

`$ ssh -i ~/mykeypair.pem -L 8888:localhost:8888 ubuntu@ec2-###-##-##-###.compute-1.amazonaws.com`

2) activate the correct virtual environment

`$ activate_env`

Note `activate_env` is an alias for `source activate tensorflow_p36`. The alias `deactivate_env` deactivates the virtual environment.

3) navigate to `cs230-project/notebooks`. Then:

`$ jupyter notebook --no-browser --port=8888`

4) the output from the above command will include a line like:

`Or copy and paste one of these URLs:`

Copy the URL below that into your browser and it will bring up a web interface to the existing notebooks. If you're creating a new notebook and want to use the same environment, be sure to choose `Environment (conda_tensorflow2_p36)`

# Contents
- classifier_hyperparam_tuning: uses Keras Tuner to identify best hyperparams (e.g., size & number of hidden layers) optimizing for performance on training set
- classifier_validation_tuning: uses Keras Tuner to identify best hyperparams (e.g., l2 and dropout) for performance on validation set; attempt to address overfitting
- classifier_validation_tuning_sigmoid: same as above using sigmoid instead of softmax for activation of output layer
- smote: creates synthetic data using approach explained [here](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
- regressor_hyperparam_tuning: same as first item listed above, but treats BDI-II prediction as regression problem instead of classification
