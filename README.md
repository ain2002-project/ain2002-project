# Stroke Prediction using Machine learning
In this collaborative coding project, we aim to develop accurate stroke prediction models using machine learning techniques. Our dataset encompasses various essential features, including age, gender, BMI, average glucose level, work type, and smoking status. To achieve reliable predictions, we perform data preprocessing, outlier detection, feature selection, and model training. Through this project, we showcase the practical application of machine learning techniques in stroke prediction, providing valuable insights for early detection and prevention. Our ultimate goal is to accurately identify individuals at risk, contributing to improved healthcare outcomes.
# 1.0 Environment Setup

## 1.1 Using local environment

### 1.1.1 Cloning the repository

First you need to clone the repository:

```bash
git clone https://github.com/ain2002-project/ain2002-project
cd ain2002-project
```

### 1.1.2 Setting up the right python version

The codes for this project has been developed and tested on Python version `3.7.12`. We have added a `.python-version` file to the repository to ensure that the correct version of Python is used. We recommend using `pyenv` to manage your Python versions. If the python `3.7.12` is installed in your system, you can skip the following steps.

Install python `3.7.12` using `pyenv`:

```bash
pyenv install 3.7.12
```

### 1.1.3 Creating a virtual environment

We should create a virtual environment so that the packages installed for this project do not interfere with the packages installed in the system. To create a virtual environment, run the following command in the root directory of the repository:

```bash
python -m venv .venv
.venv/bin/activate
```


### 1.1.4 Installing the required packages

To install the required packages, run the following command in the root directory of the repository:

```bash
pip install -r requirements.txt
```

## 1.2 On the Kaggle

You can also run the notebook on the Kaggle. The notebook is available [here](https://www.kaggle.com/osmanf/stroke-prediction-using-machine-learning).

# 2.0 Data

To run the codes on the Kaggle, you need to add [this competition dataset](https://www.kaggle.com/competitions/playground-series-s3e2/data) by kaggle and [this dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) by [fedesoriano](https://www.kaggle.com/fedesoriano) in the data section in the right panel. We have already uploaded these datasets to [our kaggle notebook](https://www.kaggle.com/osmanf/stroke-prediction-using-machine-learning).

If you want to download the data locally, you can download the datasets from kaggle by hand or you can use these commands (this will require you to be authenticated):

```bash
kaggle competitions download -c playground-series-s3e2
kaggle datasets download -d fedesoriano/stroke-prediction-dataset
```

And unzip them:

```bash
unzip stroke-prediction-dataset.zip -d data
unzip playground-series-s3e2.zip -d data/playground-series-s3e2
```

# 3.0 Running the notebook

You can run the notebook on any jupyter server (vscode, jupyterlab, by jupyter notebook command, etc.)). If you are using the local environment, you can run the notebook by running the following command in the root directory of the repository:

```bash
jupyter notebook
```

# 4.0 Training

You can run the codes as python files. They are essentially same with the notebook but with less output and no plots. You can run the codes by running the following command in the root directory of the repository:

```bash
python train.py
```

This will train and save 3 models that can be used in evaluation and inference.

# 5.0 Evaluation

The evaluation script runs evaluation metrics on the validation dataset and makes inference on the competition dataset. You can run it by

```bash
python evaluate.py
```

And if you want to see the submission score on kaggle, you can run the following command to upload the submission file:

```bash
kaggle competitions submit -c playground-series-s3e2 -f submission.csv -m "Message"
```

If everything goes well, you should get a `0.89624` private score.

# Accessing Pre-trained Models

Pretrained models will be generated and saved in `models` folder. Also we have shared the models folder in a [github release](https://github.com/osbm/ain2002-project/releases/tag/1.0).
