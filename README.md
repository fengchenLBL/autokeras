# autokeras

## AutoKeras ([autokeras.com](https://autokeras.com)): 
An AutoML system based on Keras. It is developed by [DATA Lab](http://faculty.cs.tamu.edu/xiahu/index.html) at Texas A&M University. The goal of AutoKeras is to make machine learning accessible to everyone.
### Installation (tested on Ubuntu 18 & Lawrencium Open OnDemand Cluster):
 * _Create a virtual environment, activate the virtual environment and add the virtual environment to Jupyter (recommended if using Lawrencium Open OnDemand Cluster)_ 
 ```
 python -m venv --system-site-packages ./myevn
 source ./myevn/bin/activate 
 python -m ipykernel install --user --name=myevn
 ``` 
 * Install packages on the system or within a virtual environment without affecting the system setup
 ``` 
 pip install --upgrade pip
 pip install --upgrade tensorflow
 pip install git+https://github.com/keras-team/keras-tuner.git
 pip install autokeras
 pip install plotly==4.14.3
 pip install jupyterlab "ipywidgets>=7.5"
 git clone https://github.com/fengchenLBL/autokeras.git

 ```

### Supported Tasks
AutoKeras supports several tasks with extremely simple interface:
* [Structured Data Classification](structured_data_classification.ipynb)
  * [hyperparameters](structured_data_classifier_trial.json)
* [Structured Data Regression](structured_data_regression.ipynb)
  * [hyperparameters](structured_data_regressor_trial.json)
* [Text Classification](text_classification.ipynb)
  * [hyperparameters](text_classifier_trial.json)
* [Image Classification](image_classification.ipynb)
  * [hyperparameters](image_classifier_trial.json)
* [Text Regression](text_regression.ipynb)
* [Image Regression](image_regression.ipynb)

Coming Soon: Time Series Forcasting, Object Detection, Image Segmentation.

### [Documentation](https://autokeras.com/block)

## Google Cloud AutoML
