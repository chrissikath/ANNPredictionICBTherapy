# ANNPredictionICBTherapy
All scripts related to master thesis "Prediction on Immunotherapy Response in Melanoma Patients based on Machine Learning" by Christina Kuhn April 2021

## Table of Contents
- [Abstract](#abstract)
- [Installation](#installation)
- [How to use](#howtouse)

## Abstract
In recent years, immunotherapy with immune checkpoint blockade (ICB) has shown enormous success in the treatment of melanoma. However, reliably predicting a successful therapy while avoiding therapy options without benefit at baseline is still an unsolved issue. The aim of this master thesis is therefore to define statistical models that predict the success of immune checkpoint therapies in melanoma patient cohorts using neural networks. Since resistance to ICB is related to tumor environment and host immune factors, personalized models based on a patient's genomic setup could be decisive. However, complexity and high dimensionality resulting from the transcriptome data analysed here needs to be addressed with an automated machine learning algorithm. Models were based on Artificial Neural Networks (ANN) to predict the overall and progression free survival (OS and PFS, respectively) of melanoma patients undergoing anti-CTLA4 and anti-PD1/anti-PDL1 therapy. Measures of gene expression in Transcripts per Million (TPM) from bulk tumor RNA-sequencing data were used from five melanoma datasets. Clinical variables were included such as gender, age, and the type of therapy. The ANN was then optimised to achieve the highest possible accuracy in predicting the predefined survival outcome. Problems resulting from high-dimensional data, such as overfitting, were addressed using regularization and feature selection. As a result, the ANN-based model with feature selection was shown to have the ability to predict survival (PFS) to ICB therapy with up to 86% accuracy. ANN without feature selection, however with regularization, reached up to 72% accuracy for PFS and 71% for OS, respectively. To address the problem of small patient numbers and to test reproducibility, the model was trained and validated based on the combination of all five datasets. Since the combination did not lead to an improvement in prediction, follow-up studies are necessary, whereby the developed workflow can be used as a starting point for adaption to new datasets. In summary, the developed model may contribute to personalized therapy decisions in melanoma patients

## Installation
### 1. Download Anaconda or Miniconda

(with Python 3.7)
Tested with conda version 4.10.3

### 2. Create new environment with python 3.7

```bash
conda create --name ANN_environment python=3.7
```

### 3. Activate environment

```bash
conda activate ANN_environment
``` 

### 4. Pip install all required packages

```bash
pip install -r requirements.txt
```

```bash
tensorflow==2.3.1          # Development of ANN workflow
keras==2.2.4               # Python interface for TensorFlow backend
seaborn==0.10.0            # Drawing statistical graphics like confusion matrix
scikit-learn==0.23.2       # Used for dimensionality reduction, model selection, pre-processing, and general machine learning workflow
pandas==1.1.0              # Handling datasets
imbalanced-learn==0.7.0    # Upsampling method SMOTE
matplotlib==3.0.2          # Required for seaborn
scipy==1.1.0               # Required for numerical and scientific calculations
numpy==1.16.2              # Required for numerical and scientific calculations
csv==1.0                   # Handling of CSV files
argparse==1.1              # User-friendly command-line interface
logging==0.5.1.2           # Creation of the log of the software process
```

## How to use
How to use: Mode "ann" = train model or "predict" = apply trained model to external data

ANN binary
```
python3 ANNPredictionICBTherapy.py ann # ann mode
-a ann # ann modus: basic ann worklow "ann" or ann with feature selection "fs_ann" 
-i liu_MPEST_EYA1_20_01_21.csv # input table
-o 'PFS'  # outcome feature name
-b -c 365 # if -b,then outcome = binary/classification and -c (cutoff) is requried,
# which seperated the feature into 0 and 1 
--categorical_features PFS gender LDH_Elevated Tx stage # all categorical features in input
# all clinical features which are not used for feature selection 
--clinical_features gender PFS heterogeneity nonsyn_muts stage LDH_Elevated Tx 
# the grid of parameters which should be used for fine tuning the ANN
--grid __ann_n_neurons=3 __ann_num_hidden=1 __ann_learning_rate=0.01 __ann_l1_reg=0.0 
__ann_l2_reg=0.0 __ann_dropout_rate=0.0 __ann_used_optimizer="adam","sgd" 
-s 0 # 80/20 Split state (different numbers = different 80/20 Splits) 
-m # save the best ANN model 
-u # use upsampling SMOTE-NC  
-d "2-genes-and-all-clinicals" # extra name for output directory
```

ANN mit Feature Selection
```
python3 ANNPredictionICBTherapy.py # ann modus
-a fs_ann # ann modus: nur ann "ann" oder ann mit feature selection "fs_ann" 
-i without_premed.csv # input tabelle, outcome positiion egal
-o 'PFS' # benennung der outcome variable
--categorical_features PFS gender LDH_Elevated Tx # alle features angeben, 
die kategorisch sind 
--clinical_features gender heterogeneity nonsyn_muts stage LDH_Elevated Tx 
# alle features angeben, bei der feature selection ausgeschlossen werden 
--grid __ann_n_neurons=2,4 __ann_num_hidden=1,2 __ann_learning_rate=0.01 
__ann_regularization_rate=0.2,0.4,0.6 __ann_dropout_rate=0.2,0.4,0.6 
# das grid über das das neuronale Netz läuft
-s 0 # 80/20 Split state (verschiedene Zahlen = verschiedenen 80/20 Splits)
```

Predict Mode:
```
python3 ANNPredictionICBTherapy.py predict #predict 
-j ../results/05_01_21_14_10_ann_binary/split_random_seed_0/model.json # json format
-w ../results/05_01_21_14_10_ann_binary/split_random_seed_0/model.h5 # h5 format 
-b # if outcome is binary (default = continuously)
-t liu_only_X.csv # table with features for prediction
```

