# ANNPredictionICBTherapy
All scripts related to master thesis "Prediction on Immunotherapy Response in Melanoma Patients based on Machine Learning" by Christina Kuhn April 2021


How to use: Mode "ann" = train model or "predict" = apply trained model to external data

ANN binary
```
python3 Runnable_27_01_21.py ann # ann mode
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
python3 Runnable_27_12_2020.py # ann modus
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
python3 Runnable_29_01_2021.py predict #predict 
-j ../results/05_01_21_14_10_ann_binary/split_random_seed_0/model.json # json format
-w ../results/05_01_21_14_10_ann_binary/split_random_seed_0/model.h5 # h5 format 
-b # if outcome is binary (default = continuously)
-t liu_only_X.csv # table with features for prediction
```

