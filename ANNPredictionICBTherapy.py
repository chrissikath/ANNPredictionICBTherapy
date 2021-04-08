#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[12]:


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import math
import csv
import sklearn
import time
import random
import sys
import os
import argparse
import logging
import tensorflow.keras.backend as K
from imblearn.over_sampling import SMOTENC
from datetime import datetime
from matplotlib import pyplot as plt
from tensorflow import keras
from scipy import stats
from sklearn import svm, datasets, metrics, model_selection, svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedKFold, KFold, ShuffleSplit, StratifiedShuffleSplit
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.datasets import load_boston
from sklearn.linear_model import ElasticNetCV, ElasticNet, LogisticRegression, LogisticRegressionCV


# # Funktionen

# In[77]:


# Funktionen
def load_data(file):
    """
    Load csv file 
    Return: DataFrame 
    """
    print("-----Load Dataset-----")
    dataset = pd.read_csv(file)
    return dataset


def choose_input_columns(dataset, genes, clincial_variables, tpm_removal):
    """
    Choose number of genes or specific genes ["MET","EYA1"]
    and clinical variables from dataset
    Return: dataset_short (genes + clinical variables) 
    """
    print()
    print("-----Choose input coloumns-----")
    if isinstance(genes, int):  # if genes == int: take all genes
        print(str(genes) + " genes taken.")
        genes = dataset.iloc[:, 1:genes]
    elif str(genes) == '':  # if no genesfe
        pass
    else:  # take specific genes
        print(str(genes) + " taken.")
        genes = dataset[genes]

    if tpm_removal == True:
        print("-----Remove all Genes TPM <2-----")
        print("Number Genes: ", genes.shape[1])
        obj = (genes < 2).all()
        for key, value in obj.iteritems():
            if value == True:
                genes = genes.drop(columns=[key])
        print("Number Genes after removed: ", genes.shape[1])
    print()

    dataset_variables = dataset[clincial_variables]
    print("Clinical variables choosen: ")
    print(dataset_variables.columns)
    print()

    if str(genes) == '':  # if no genes taken
        dataset_short = dataset_variables
    else:   # append genes with clinical variables
        dataset_short = pd.concat([genes, dataset_variables], axis=1)

    print("-----Remove NAs-----")
    print("With NAs: ", dataset_short.shape[0])
    dataset_short = dataset_short.dropna()
    print("Removed NA: ", dataset_short.shape[0])
    print()
    return dataset_short


def choose_input_columns_with_genes_return(dataset, number_genes, clincial_variables):
    """
    Same as choose_input_columns but also returns genes and dataset_variables seperatly
    Use: merge gene set 
    Return: (dataset_short, genes, dataset_variables)
    """
    print()
    print("-----Choose input coloumns-----")
    if isinstance(genes, int):  # if genes == int: take all genes
        print(str(genes) + " genes taken.")
        genes = dataset.iloc[:, 1:genes]
    elif str(genes) == '':  # if no genes
        pass
    else:  # take specific genes
        print(str(genes) + " taken.")
        genes = dataset[genes]

    dataset_variables = dataset[clincial_variables]
    print("Clinical variables choosen: ")
    print(dataset_variables.columns)

    if str(genes) == '':  # if no genes taken
        dataset_short = dataset_variables
    else:   # append genes with clinical variables
        dataset_short = pd.concat([genes, dataset_variables], axis=1)
    return(dataset_short, genes, dataset_variables)


def get_merge_genes(genes_Gide, genes_Liu):
    """
    Merge two different gene set by names
    Return: merged_genes 
    """
    print()
    print("-----Get merged genes-----")
    merge_genes = []
    for gene in genes_Gide.columns.tolist():
        if gene in genes_Liu.columns.tolist():
            merge_genes.append(gene)
    print("Number merged genes: ", len(merge_genes))
    return merge_genes


def prepare_dataset(liu_or_gide, dataset, bin_categorical, string_to_binary, continous_to_binary, target):
    """
    Prepare dataset 
    -> Features and Target structure 
    -> categorical values to binary (string_to_binary) gender (M,F) to 1,0 
    -> continuous target to binary (continous_to_binary) OS to 1,0
    -> binary string to binary value (bin_categorical) Tx to 1,0 
    Return: (X,y)
    """
    logging.info("------Prepare dataset: --------")
    logging.info("Bin_categorical: {}".format(bin_categorical))
    logging.info("string_to_binary: {}".format(string_to_binary))
    logging.info("continous_to_binary: {}".format(continous_to_binary))
    logging.info("Target: {}".format(target))

    # make OS:730 to 0 or 1
    for key, value in continous_to_binary.items():
        dataset[key] = np.where(
            dataset[key] >= value, 1, 0)

    # make gender M = 1 or F = 0
    if string_to_binary != "":
        value = string_to_binary
        dataset[value] = np.where(dataset[value] == "M", 1, 0)

    # make any other categorical value (Tx) to 1,0
    if len(bin_categorical) != 0:
        for elem in bin_categorical:
            label = LabelEncoder()
            dataset[elem] = label.fit_transform(dataset[elem])
            
    X = dataset.drop(labels=[target], axis=1)
    y = dataset[target]
    logging.info(X.head())
    logging.info(y.head())

    return (X, y)


def grid_search_and_val(directory, X_train, y_train, X_test, y_test, model_type, param_grid, save_model, batch_size,
                        earlystopping_patience, n_features, epochs=500):
    """
    Grid SearchCV with train dataset and Validation with test datatset
    Print Best Params
    Print Confusion Matrix Results 

    Return: np.mean(scores), np.mean(sens_list), np.mean(spec_list), np.mean(ppv_list), np.mean(npv_list)
    """
    print()
    print("-----GridSearch------")
    logging.info("-----GridSearch------")
    logging.info("Used param grid: {}".format(param_grid))

    f = open(directory+"/"+"param_grid.txt", "a+")
    for key in param_grid:
        f.write(key+":"+str(param_grid[key])+"\n")

    f = open(directory+"/"+"used_params.txt", "w+")
    f.write("epochs: " + str(epochs) + "\n" + "batch size: " + str(batch_size) + "\n" +
            "earlystopping: " + str(earlystopping_patience) + "\n" + "N_features: " +
            str(n_features) + " (Only with Feature Selection)")
    f.close()

    # Early stopping criteria
#     early_stopping_cb = kecras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=earlystopping_patience, mode='auto')

    learning_rate = tf.keras.callbacks.LearningRateScheduler(step_decay)

    callbacks = [reduce_lr, early_stop]

    # Three different types of ANNs (binary, continuously, autoencoder)
    if model_type == "binary":
        print("Binary Model")
        logging.info("Binary Model")
        ann = KerasClassifier(build_fn=binaryModel, input_shape=X_train.shape[1],
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.2)
    elif model_type == "continuously":
        print("Continuously Model")
        logging.info("Continuously Model")
        ann = KerasRegressor(build_fn=continuouslyModel, input_shape=X_train.shape[1],
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_split=0.2)
    elif model_type == "autoencoder":
        print("Autoencoder Model")
        logging.info("Autoencoder Model")
        ann = KerasRegressor(build_fn=binaryModelAutoencoder, input_shape=X_train.shape[1],
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_split=0.2)

    # Pipeline for GridSearchCV (first StandardScaler then apply ann)
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('ann', ann)])
    grid_search_result = GridSearchCV(
        pipe, param_grid=param_grid, n_jobs=1, cv=5)
    grid_search_result = grid_search_result.fit(X_train, y_train,
                                                ann__callbacks=callbacks, ann__verbose=0)

    print()
    print("-----Grid Search Result-----")
    logging.info("-----Grid Search Result-----")
    logging.info("Best: %f using %s" %
                 (grid_search_result.best_score_, grid_search_result.best_params_))
    print("Best: %f using %s" %
          (grid_search_result.best_score_, grid_search_result.best_params_))

    f = open(directory+"/"+"best_params.txt", "a+")
    for key in grid_search_result.best_params_:
        f.write(key+":"+str(grid_search_result.best_params_[key])+"\n")
    f.close()

    print()
    print("-----Validation with test set-----")
    logging.info("-----Validation with test set-----")
    # Scale train data and transform train data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == "continuously":
        mae_scores = []
        r2_scores = []
        rmse_scores = []
        for i in range(30):
            print("Validation-Run: ", i)
            model = continuouslyModel(
                input_shape=X_train.shape[1],
                n_neurons=grid_search_result.best_params_['ann__n_neurons'],
                learning_rate=grid_search_result.best_params_[
                    'ann__learning_rate'],
                used_optimizer =  grid_search_result.best_params_['ann__used_optimizer'],
                l1_reg=grid_search_result.best_params_['ann__l1_reg'],
                l2_reg=grid_search_result.best_params_['ann__l2_reg'],
                num_hidden=grid_search_result.best_params_['ann__num_hidden'],
                dropout_rate=grid_search_result.best_params_['ann__dropout_rate'])

            history = model.fit(X_train, y_train, callbacks=callbacks, shuffle=True,
                                batch_size=batch_size, validation_split=0.2, epochs=epochs, verbose=0)

            test_loss, test_mae = model.evaluate(X_test, y_test)

            if not os.path.exists(directory+"/iteration_"+str(i)):
                os.makedirs(directory+"/iteration_"+str(i))

            plt.clf()
            for j in range(0, int(len(history.history.keys())/2)):
                plt.plot(history.history[list(history.history.keys())[j]])
                plt.plot(history.history['val_' +
                                         list(history.history.keys())[j]])
                plt.title('model ' + list(history.history.keys())[j])
                plt.ylabel(list(history.history.keys())[j])
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                plt.savefig(directory+'/iteration_'+str(i)+"/" +
                            str(list(history.history)[j])+'.png')
                plt.clf()

            # predict
            y_predicted = model.predict(X_test)

            count = 0
            for i in y_test:
                print(i, y_predicted[count])
                count += 1

            rmse = (np.sqrt(mean_squared_error(y_test, y_predicted)))
            r_squared = r2_score(y_test, y_predicted)

            # append in lists
            mae_scores.append(test_mae)
            rmse_scores.append(rmse)
            r2_scores.append(r_squared)

        if save_model == True:
            model_json = model.to_json()
            with open(directory+"/"+"model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            ann_val.save_weights(directory+"/"+"model.h5")
            loggin.info("Saved model to disk")

        f = open(directory+"/"+"metrics.txt", "a+")
        f.write(
            "MAE {0:.2f} +/-({1:.2f}) \n".format(np.mean(mae_scores),  np.std(mae_scores)))
        f.write(
            "RMSE {0:.2f} +/-({1:.2f}) \n".format(np.mean(rmse_scores),  np.std(rmse_scores)))
        f.write(
            "R^2 {0:.2f} +/-({1:.2f}) \n".format(np.mean(r2_scores),  np.std(r2_scores)))
        f.close()

        print(
            "MAE {0:.2f} +/-({1:.2f}) ".format(np.mean(mae_scores),  np.std(mae_scores)))
        print(
            "RMSE {0:.2f} +/-({1:.2f}) ".format(np.mean(rmse_scores),  np.std(rmse_scores)))
        print(
            "R^2 {0:.2f} +/-({1:.2f}) ".format(np.mean(r2_scores),  np.std(r2_scores)))
    else:
        scores = []
        sens_list = []
        spec_list = []
        ppv_list = []
        npv_list = []
        for i in range(30):  # Validation with test dataset (30 times)
            print("Validation-Run: ", i)
            # define best params from GridSearch
            if model_type == "binary":
                ann_val = binaryModel(input_shape=X_train.shape[1], 
                                      n_neurons=grid_search_result.best_params_['ann__n_neurons'],
                                      learning_rate=grid_search_result.best_params_['ann__learning_rate'],
                                     used_optimizer =  grid_search_result.best_params_['ann__used_optimizer'],
                                      l1_reg=grid_search_result.best_params_['ann__l1_reg'],
                                    l2_reg=grid_search_result.best_params_['ann__l2_reg'],
                                    num_hidden=grid_search_result.best_params_['ann__num_hidden'],
                                    dropout_rate=grid_search_result.best_params_['ann__dropout_rate'])
                
                #(input_shape, n_neurons, num_hidden, learning_rate, optimizer, l1_reg, l2_reg, dropout_rate):
            elif model_type == "autoencoder":
                ann_val = binaryModelAutoencoder(input_shape=X_train.shape[1], n_neurons=grid_search_result.best_params_['ann__n_neurons'],
                                                 learning_rate=grid_search_result.best_params_[
                    'ann__learning_rate'],optimizer = grid_search_result.best_parms_['ann__optimizer'],
                    l1_reg=grid_search_result.best_params_[
                    'ann__l1_reg'],
                    l2_reg=grid_search_result.best_params_[
                    'ann__l2_reg'],
                    num_hidden=grid_search_result.best_params_[
                        'ann__num_hidden'],
                    dropout_rate=grid_search_result.best_params_['ann__dropout_rate'])

            # fit ann with best params
            history = ann_val.fit(X_train, y_train, batch_size=batch_size, verbose=0,
                                  validation_split=0.2, shuffle=True, callbacks=callbacks, epochs=epochs)

            if not os.path.exists(directory+"/iteration_"+str(i)):
                os.makedirs(directory+"/iteration_"+str(i))

            # plot training history
            plt.clf()
            for j in range(0, int(len(history.history.keys())/2)):
                plt.plot(history.history[list(history.history.keys())[j]])
                plt.plot(history.history['val_' +
                                         list(history.history.keys())[j]])
                plt.title('model ' + list(history.history.keys())[j])
                plt.ylabel(list(history.history.keys())[j])
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                plt.savefig(directory+'/iteration_'+str(i)+"/" +
                            str(list(history.history)[j])+'.png')
                plt.close()

            # evaluate with test set
            test_loss, test_acc = ann_val.evaluate(X_test, y_test, verbose=0)
            y_predicted = (ann_val.predict(X_test) > 0.5).astype("int32")
            print(y_predicted)
            count = 0
            for test_object in y_test:
                print(test_object, y_predicted[count])
                count += 1

            sens, spec, ppv, npv = sens_spec_ppn_npv(
                y_test, y_predicted)  # calculate confusion matrix
            scores.append(test_acc)
            sens_list.append(sens)
            spec_list.append(spec)
            ppv_list.append(ppv)
            npv_list.append(npv)

            # plot confusion matrix of last validation run
            fig = plt.figure(figsize=(4, 4))
            sns.heatmap(confusion_matrix(y_test, y_predicted, labels=[0,1]),
                        vmax=.8, square=True, annot=True)
            plt.show()
            plt.savefig(directory+'/iteration_'+str(i) +
                        "/"+'confusion_matrix.png')
            plt.close()

        plt.clf()
        # plot confusion matrix of last validation run
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix(y_test, y_predicted, labels=[0,1]),
                    vmax=.8, square=True, annot=True)
        plt.show()
        plt.savefig(directory+"/"+'last_run_confusion_matrix.png')
        plt.close()

        if save_model == True:
            model_json = ann_val.to_json()
            with open(directory+"/"+"model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            ann_val.save_weights(directory+"/"+"model.h5")
            logging.info("Saved model to disk")

        f = open(directory+"/"+"metrics.txt", "a+")
        f.write(
            "Accuracy {0:.2f} +/-({1:.2f}) \n".format(np.mean(scores),  np.std(scores)))
        f.write(
            "Sensitivity {0:.2f} +/-({1:.2f}) \n".format(np.mean(sens_list),  np.std(sens_list)))
        f.write(
            "Specificity {0:.2f} +/-({1:.2f}) \n".format(np.mean(spec_list),  np.std(spec_list)))
        f.write(
            "PPV {0:.2f} +/-({1:.2f}) \n".format(np.mean(ppv_list),  np.std(ppv_list)))
        f.write(
            "NPV {0:.2f} +/-({1:.2f}) \n".format(np.mean(npv_list),  np.std(npv_list)))
        f.close()

        # print Results
        print(
            "Accuracy {0:.2f} +/-({1:.2f}) ".format(np.mean(scores),  np.std(scores)))
        print(
            "Sensitivity {0:.2f} +/-({1:.2f}) ".format(np.mean(sens_list),  np.std(sens_list)))
        print(
            "Specificity {0:.2f} +/-({1:.2f}) ".format(np.mean(spec_list),  np.std(spec_list)))
        print(
            "PPV {0:.2f} +/-({1:.2f}) ".format(np.mean(ppv_list),  np.std(ppv_list)))
        print(
            "NPV {0:.2f} +/-({1:.2f}) ".format(np.mean(npv_list),  np.std(npv_list)))

    logging.info("-----ANN and Validation Done-----")


def binaryModel(input_shape, n_neurons, num_hidden, learning_rate, used_optimizer, l1_reg, l2_reg, dropout_rate):
    '''
    Binary classification ANN Model with dropout layer 
    Optimize: 
    -> number neurons (n_neurons)
    -> learning rate adam (learning_rate)
    -> dropout rate (dropout_rate)

    Return: Model 
    '''
    inputs = Input((input_shape,))
    for i in range(num_hidden):
        hidden = Dense(n_neurons, activation="elu",
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(inputs)
        dropout = Dropout(dropout_rate)(hidden)
    
    outputs = (Dense(1, activation="sigmoid"))(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    if used_optimizer == "sgd":
        ann_optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.5, nesterov=True)
        model.compile(loss="binary_crossentropy",
                  optimizer=ann_optimizer, metrics=['accuracy'])
        
    elif used_optimizer == "adam":
        ann_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss="binary_crossentropy",
                  optimizer=ann_optimizer, metrics=['accuracy'])
    
    # summarize layers
    print(model.summary())
    # plot graph
    plot_model(model, to_file='multilayer_perceptron_graph_binary.png')

    return model


def binaryModelAutoencoder(input_shape, n_neurons, num_hidden, learning_rate, regularization_rate, dropout_rate):
    '''
    Binary classification ANN Model with dropout layer -> Autoencoder
    Optimize: 
    -> number neurons (n_neurons)
    -> learning rate adam (learning_rate)
    -> dropout rate (dropout_rate)

    Return: Model 
    '''
    model = Sequential()
    model.add(Dense(input_shape, activation="elu",
                    input_dim=input_shape))  # First Layer

    model.add(Dense(265, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))

#     for i in range(num_hidden):  # number hidden layers
#         model.add(Dense(n_neurons, activation="relu",
#                         kernel_regularizer=tf.keras.regularizers.l1(regularization_rate)))
#         model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="sigmoid"))
    #optimizer = keras.optimizers.SGD(learning_rate=learning_rate , momentum =0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer, metrics=['accuracy'])
    return model


def continuouslyModel(input_shape, n_neurons, num_hidden, learning_rate, used_optimizer, l1_reg, l2_reg, dropout_rate):
    '''
    Regression ANN Model with l1 regularization
    Optimize: 
    -> number neurons (n_neurons)
    -> learning rate adam (learning_rate)
    -> regularizaion rate l1 (regularization_rate)

    Metrics: accuracy, rmse, r2_keras

    Return: Model 
    '''
    inputs = Input((input_shape,))
    for i in range(num_hidden):
        hidden = Dense(n_neurons, activation="elu",
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(inputs)
        dropout = Dropout(dropout_rate)(hidden)
    
    outputs = (Dense(1, activation="linear"))(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    if used_optimizer == "sgd":
        ann_optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.5, nesterov=True, clipnorm=1.)
        model.compile(loss="mse", optimizer=ann_optimizer,
                  metrics=['mae'])
        
    elif used_optimizer == "adam":
        ann_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.)
        model.compile(loss="mse", optimizer=ann_optimizer,
                  metrics=['mae'])

    # summarize layers
    print(model.summary())
    # plot graph
    plot_model(model, to_file='multilayer_perceptron_graph_continuously.png')
    
    return model


def feature_Selection(X, y, n_features=150, step=100, verbose=1):
    """
    Feature Selection withs SVM linear kernel (step = 100)
    Variabel Step, verbose and n_features 

    Input: 
    -> X = features
    -> y = target
    -> n_features = Number features to choose 

    Return: (selector, selected X, columns, ranks)
    """
    print()
    print("-----Feature Selection-----")
    estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features,
                   step=step, verbose=verbose)
    X_selected = selector.fit_transform(X, y)

    columns = []
    ranks = {}
    for i in range(X.shape[1]):
        if selector.support_[i] == True:
            columns.append(i)
        d = {str(selector.ranking_[i]): str(i)}
        ranks.update(d)
        #print('Column: %d, Selected %s, Rank: %.3f' % (i, selector.support_[i], selector.ranking_[i]))

    print("-----Feature Selection DONE------")
    return (selector, X_selected, columns, ranks)


def featureSelection_selector(X, y, n_features=150):
    """
    Feature Selection withs SVM linear kernel (step = 100)
    Use: train on Liu, transform on Gide 

    Input: 
    -> X = features
    -> y = target
    -> n_features = Number features to choose 

    Return: selector 
    """
    print()
    print("-----Feature Selection-----")
    estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features,
                   step=100, verbose=1)
    X = selector.fit_transform(X, y)
    return selector


def crossValidationContinuously(X_train, y_train, X_test, y_test, best_learning_rate, best_n_neurons, epochs=300, batch_size=32):
    """
    Cross Validation for Continuously Model
    """
    print()
    print("-----Cross Validation-----")
    scores = []
    msescores = []
    rsscores = []
    rs_score = []
    for i in range(30):
        print("Validation-Run: ", i)
        model = continuouslyModel(
            input_shape=X_train.shape[1], n_neurons=best_n_neurons, learning_rate=best_learning_rate)
        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50)
        history = model.fit(X_train, y_train, batch_size=batch_size, callbacks=[
                            early_stopping_cb], validation_split=0.2, epochs=epochs, verbose=0)
        test_loss, test_rmse, test_rsquared = model.evaluate(X_test, y_test)
        # predict
        y_predicted = model.predict_classes(X_test)
        mae = mean_absolute_error(y_test, y_predicted)
        r_squared = r2_score(y_test, y_predicted)
        # append in lists
        scores.append(test_rmse)
        msescores.append(mae)
        rsscores.append(r_squared)
        rs_score.append(test_rsquared)

    plotHistory(history)
    # Beispiel des letzten Durchlaufs
    #fig = plt.figure(figsize = (4,4))
    #sns.heatmap(confusion_matrix(y_test, y_predicted), vmax = .8, square = True, annot=True)
    # plt.show()

    print("RMSE %.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    print("MAE %.2f%% (+/- %.2f%%)" % (np.mean(msescores), np.std(msescores)))
    print("R Squared %.2f%% (+/- %.2f%%)" %
          (np.mean(rsscores), np.std(rsscores)))
    print("R Squared metric %.2f%% (+/- %.2f%%)" %
          (np.mean(rs_score), np.std(rs_score)))


def sens_spec_ppn_npv(y_true, y_pred):
    """
    Calculate sensitivity, specificity, PPV, NPV

    Params: 
    y_pred - Predicted labels
    y_true - True labels 

    Returns:  (sensitivity, specificity, PPV, NPV)
    """
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    ACC = (TP+TN)/(TP+FP+FN+TN)

    return (sensitivity, specificity, PPV, NPV)


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def clean_out_Gide(dataset_Gide):
    """
    Clean out Gide dataset 
    -> drop NAs
    -> only use PD1
    -> only use PRE
    -> only use non BRAF mutation
    """
    print()
    print("----Clean Out Datasets-----")
    print("Gide")
    # Prepare Gide
    indexNames = dataset_Gide[dataset_Gide['Treatment_'] != "PD1"].index
    # Delete rowse where Treatment is not PD1
    dataset_Gide.drop(indexNames, inplace=True)
    print(len(dataset_Gide), " nur PD1")

    indexNames = dataset_Gide[dataset_Gide['RNAseq'] != "PRE"].index
    # Delete rowse where RNASeq is not PRE
    dataset_Gide.drop(indexNames, inplace=True)
    print(len(dataset_Gide), " nur PRE")

    indexNames = dataset_Gide[dataset_Gide['BRAF_V600_mutation'] != 0].index
    dataset_Gide.drop(indexNames, inplace=True)
    print(len(dataset_Gide), " ohne BRAF")

    return dataset_Gide


def clean_out_Liu(dataset_Liu):
    """
    Clean out Liu dataset 
    -> drop NAs
    -> only use numPrior == 0
    """
    print()
    print("----Clean Out Dataset-----")
    # Prepare Liu
    print("Liu")

    #indexNames = dataset_Liu[ dataset_Liu['numPriorTherapies'] !=  0].index
    #dataset_Liu.drop(indexNames , inplace=True)
    #print(len(dataset_Liu), "ohne post Therapien")
    return dataset_Liu


def clean_out_Riaz(dataset_Riaz):
    """
    Clean out Riaz dataset 
    -> drop NAs
    -> only use PRE
    -> change os_week to os_days 
    """
    print("Riaz")
    indexNames = dataset_Riaz[dataset_Riaz['On_Pre'] != "Pre"].index
    dataset_Riaz.drop(indexNames, inplace=True)
    print(len(dataset_Riaz), "ohne On_Pre")

    print("change weeks to days")
    # os_weeks in days umrechnen
    dataset_Riaz['os_weeks'] = dataset_Riaz['os_weeks']*7

    return dataset_Riaz


def overview_data(dataset, clinical):
    """
    Get overview of data
    -> Summary 
    -> plot clinical values 
    -> plot correlations of clinical values 
    """
    print(dataset.describe())
    shortDataset = dataset[clinical]

    shortDataset.hist(figsize=(12, 10))
    plt.show()

    C_mat = shortDataset.corr()
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(C_mat, vmax=.8, square=True, annot=True)
    plt.show()


def plotHistory(history):
    """
    Plot training history of ANN
    """
    for i in range(0, int(len(history.history.keys())/2)):
        plt.plot(history.history[list(history.history.keys())[i]])
        plt.plot(history.history['val_' + list(history.history.keys())[i]])
        plt.title('model ' + list(history.history.keys())[i])
        plt.ylabel(list(history.history.keys())[i])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def rmse(y_true, y_pred):
    """
    RMSE metric function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def r2_keras(y_true, y_pred):
    """
    r2 metric function 
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def drop_outliers(dataset, features):
    """
    Drop Outliers for features where z value > 3
    Return dataset without outliers 
    """
#     z = np.abs(stats.zscore(dataset))
#     outliers = (np.where(z > 3))
#     print(outliers )

    print("------Delete outliers------")
    for feature in features:
        print(feature)
        # z score for this feature
        z_scores = np.abs(stats.zscore(dataset[feature]))

        outliers = (np.where(z_scores > 3, True, False))
        print(np.where(outliers == True)[0])
        for i in (np.where(outliers == True)[0]):
            try:
                dataset = dataset.drop(i)
                dataset = dataset.reset_index(drop=True)
            except:
                print("Row already dropped.")
    return dataset


def feature_selection_logreg(param_grid, X, X_train, y_train, X_test, y_test):
    print()
    print("------Feature Selection via LogReg ElasticNet-------")
    print("------LogReg ElasticNet Tuning-------")
    # LogisticRegression Model with elastic net
    log_reg = LogisticRegression(
        penalty='elasticnet', solver='saga', max_iter=5000, verbose=2, n_jobs=-1)

    grid_search_result = GridSearchCV(
        log_reg, param_grid=param_grid, n_jobs=1, cv=2)
    grid_search_result = grid_search_result.fit(
        X_train, y_train)

    print("-----Grid Search Result-----")
    print("Best: %f using %s" %
          (grid_search_result.best_score_, grid_search_result.best_params_))

    print("------Feature Selection-------")
    # LogisticRegression with best params
    log_reg_best_param = LogisticRegression(
        penalty='elasticnet', solver='saga', max_iter=5000, verbose=2, n_jobs=-1,
        C=grid_search_result.best_params_['C'],
        l1_ratio=grid_search_result.best_params_['l1_ratio'])

    log_reg_best_param.fit(X_train, y_train)
    y_predicted = log_reg_best_param.predict(X_test)
    print("Score Test Dataset:")
    print(log_reg_best_param.score(X_test, y_test))
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix(y_test, y_predicted),
                vmax=.8, square=True, annot=True)
    plt.show()

    # Return short X_train, X_test based on selected features
    selected_coeff = (
        np.abs(np.round(log_reg_best_param.coef_, decimals=2)) > 0)[0]

    print("Länge selected coeff", len(selected_coeff))
    print("Länge X_train", X_train.shape)
    print(type(selected_coeff))

    # select from dataframe by boolean list
    X_selected_train = X_train[:, selected_coeff]
    X_selected_test = X_test[:, selected_coeff]

    print("Number selected coefficients", len(X_selected_train[0]))

    odds = np.exp(log_reg_best_param.coef_[0])
    not_odds = (log_reg_best_param.coef_[0])

    coefficients = pd.DataFrame(not_odds,
                                X.columns,
                                columns=['coef']).sort_values(by='coef', ascending=False)

    coefficient_names = X.columns[selected_coeff]
    print("Features selected ", coefficient_names)
    coefficients_with_value = coefficients.loc[coefficient_names, ]

    return (coefficients_with_value, X_selected_train, X_selected_test)


def feature_selection_svc(param_grid, binary, X, X_train, y_train, X_test, y_test, n_features, step=100, verbose=1):
    print()
    print("------Feature Selection via SVC-------")
    logging.info("------Feature Selection via SVC-------")
    logging.info("------SVC Fine Tuning-------")
    if binary == True:
        clf = svm.SVC()
    else:
        clf = svm.SVR()

    grid_search_result = GridSearchCV(
        clf, param_grid=param_grid, n_jobs=1, cv=3)
    grid_search_result = grid_search_result.fit(
        X_train, y_train)

    logging.info("-----Grid Search Result-----")
    logging.info("Best: %f using %s" %
                 (grid_search_result.best_score_, grid_search_result.best_params_))

    # LogisticRegression with best params
    if binary == True:
        clf_bestparams = svm.SVC(
            kernel=grid_search_result.best_params_['kernel'],
            C=grid_search_result.best_params_['C'])
    else:
        clf_bestparams = svm.SVR(
            kernel=grid_search_result.best_params_['kernel'],
            C=grid_search_result.best_params_['C'])

    clf_bestparams.fit(X_train, y_train)
    y_predicted = clf_bestparams.predict(X_test)
    logging.info("Score Test Dataset: {}".format(
        clf_bestparams.score(X_test, y_test)))
    if binary == True:
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix(y_test, y_predicted),
                    vmax=.8, square=True, annot=True)
        plt.show()
    else:
        rmse = (np.sqrt(mean_squared_error(y_test, y_predicted)))
        r_squared = r2_score(y_test, y_predicted)
        logging.info("RMSE: {}".format(rmse))
        logging.info("R Squared: {}".format(r_squared))

    logging.info("------Feature Selection------")
    if binary == True:
        estimator = SVC(kernel=grid_search_result.best_params_[
                        'kernel'], C=grid_search_result.best_params_['C'])
    else:
        estimator = SVR(kernel=grid_search_result.best_params_[
                        'kernel'], C=grid_search_result.best_params_['C'])

    selector = RFE(estimator, n_features_to_select=n_features,
                   step=step, verbose=verbose)
    selector = selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    columns = []
    logging.info(selector.support_)
    for i in range(X_train.shape[1]):
        if selector.support_[i] == True:
            columns.append(i)

    logging.info("-----Feature Selection DONE------")
    coefficients = X.columns[columns]

    return (coefficients, selector, X_train_selected, X_test_selected)


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    learning_rate = initial_lrate * math.pow(drop,
                                             math.floor((1+epoch)/epochs_drop))
    return learning_rate


def load_model_and_predict(model_json, model_h5, binary, table):
    """
    Takes the model in json and h5 format and predicts the outcome for given table 
    Params:
    model_json - model in json format
    model_h5 - model in h5 format
    binary - True for binary outcome, False for continuously outcome 
    table - Predict outcome for given table
    """
    # load json and create model
    data = load_data(table)
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_h5)
    print("Loaded model from disk")
    if binary == True:
        predicted_y = loaded_model.predict_classes(data)
        print(predicted_y)
    else:
        predicted_y = loaded_model.predict(data)
        print(predicted_y)


# # Ohne FS

# ## Binary

# In[17]:


def ann_binary(directory, dataset, outcome, cutoff, categorical_features, 
                       smote_upsample, grid, model_type, split_state, save_model, batch_size,earlystopping_patience):
    """
    Ann with binary outcome 
    with the availabily of upsample via SMOTE-NC
    """
    
    continous_to_binary_liu = {outcome: cutoff} 
    X, y = prepare_dataset("liu", dataset, categorical_features,
                           "", continous_to_binary_liu, outcome)

    #split in train test data 80/20 split stratified and shuffled 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state = split_state, test_size = 0.2, stratify = y, shuffle = True)
    
    if smote_upsample == True:
        logging.info("Upsample less represented class.")
        #For SMOTE-NC we need to pinpoint the column position where is the categorical features are.
        categorical_f = []
        categorical_features.remove(outcome)
        for feature in categorical_features:
            categorical_f.append(X_train.columns.get_loc(feature))
        #For SMOTE-NC we need to pinpoint the column position where is the categorical features are.
        smotenc = SMOTENC(categorical_features=categorical_f,random_state = 101)
        X_train, y_train = smotenc.fit_resample(X_train, y_train)
    
    #create for each split a new directory 
    new_dir = directory + "/" + "split_random_seed_"+ str(split_state)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    log = open(new_dir+"/"+"myprog.log", "a+")
    sys.stdout = log

    if model_type == "binary":
        grid_search_and_val(new_dir, X_train, y_train, X_test, y_test, 'binary', grid, save_model, batch_size, 
                            earlystopping_patience, n_features=0, epochs=1000)
    elif model_type == "autoencoder":
        grid_search_and_val(new_dir, X_train, y_train, X_test, y_test, 'autoencoder', grid, save_model,batch_size, 
                            earlystopping_patience,n_features=0,epochs=100)


# ## Continuously

# In[6]:


def ann_conti(directory, dataset, outcome, cutoff, categorical_features, 
                       smote_upsample, grid, split_state, save_model,batch_size, earlystopping_patience):
    """
    Ann with continuously outcome 
    with the availabily of upsample via SMOTE-NC
    """
    
    continous_to_binary_liu = {}

    X, y = prepare_dataset("liu", dataset, categorical_features,
                           "", continous_to_binary_liu, outcome)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=split_state, test_size=0.2, shuffle=True)
    
    #create for each split a new directory 
    new_dir = directory + "/" + "split_random_seed_"+ str(split_state)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        
    log = open(new_dir+"/"+"myprog.log", "a+")
    sys.stdout = log

    grid_search_and_val(new_dir, X_train, y_train, X_test, y_test, 'continuously',
                        grid, save_model, batch_size, earlystopping_patience ,n_features=0, epochs=1000)


# # Mit FS (SVM)

# ## Binary

# In[9]:


def ann_fs_binary(directory, dataset, outcome, cutoff, categorical_features, clinical_features,
                       smote_upsample, grid, split_state,save_model,n_features, batch_size, earlystopping_patience):
    """
    Ann with feature selection (SVC) with binary outcome 
    with the availabily of upsample via SMOTE-NC
    """

    continous_to_binary_liu = {outcome: cutoff}
    data_for_svm =dataset.copy()

    X, y = prepare_dataset("liu", data_for_svm, categorical_features,
                           "", continous_to_binary_liu, outcome)

    X_genes = X.drop(columns=clinical_features)
    X_train, X_test, y_train, y_test = train_test_split(
         X_genes, y, random_state=split_state, test_size=0.2, stratify=y)


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # mit SVC
    param_grid_SVC = dict(C=[0.1,0.2,0.4,0.4,0.6,0.7,0.8,0.9,0.99],
                      kernel= ['linear'])
    svm_features, selector, X_train_selected, X_test_selected = feature_selection_svc(
        param_grid_SVC, True, X, X_train, y_train, X_test, y_test, n_features, verbose=0)

    ############################################################################################

    X, y = prepare_dataset("liu", dataset, categorical_features,
                           "", continous_to_binary_liu, outcome)
    X_svm = X[svm_features]
    X_new = pd.concat([X_svm, X[clinical_features]], axis=1)
    logging.info(svm_features)
    
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=split_state, test_size=0.2, shuffle=True, stratify = y)

    if smote_upsample == True:
        logging.info("Upsample less represented class.")
        #For SMOTE-NC we need to pinpoint the column position where is the categorical features are.
        categorical_f = []
        categorical_features.remove(outcome)
        for feature in categorical_features:
            categorical_f.append(X_new.columns.get_loc(feature))
        smotenc = SMOTENC(categorical_features=categorical_f,random_state = 101)
        X_train, y_train = smotenc.fit_resample(X_train, y_train)
    
    new_dir = directory + "/" + "split_random_seed_"+ str(split_state)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        
    f = open(new_dir+"/"+"features.txt", "w+")
    for i in svm_features:
        f.write(str(i)+"\n")
    f.close()
        
    log = open(new_dir+"/"+"myprog.log", "a+")
    sys.stdout = log
        
    grid_search_and_val(new_dir, X_train, y_train, X_test, y_test, 'binary',
                            grid,save_model, batch_size, earlystopping_patience, n_features, epochs=1000)


# ## Continuously

# In[14]:


def ann_fs_conti(directory, dataset, outcome, cutoff, categorical_features, clinical_features,
                       smote_upsample, grid, split_state, save_model, n_features, batch_size, earlystopping_patience):
    """
    Ann with feature selection (SVR) with continuously outcome 
    """
    
    continous_to_binary_liu = {}
    data_for_svm =dataset.copy()

    X, y = prepare_dataset("liu", data_for_svm, categorical_features,
                           "", continous_to_binary_liu, outcome)

    X_genes = X.drop(columns=clinical_features)
    X_train, X_test, y_train, y_test = train_test_split(
         X_genes, y, random_state=split_state, test_size=0.2)


    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # mit SVC
    param_grid_SVC = dict(C=[0.1,0.2,0.4,0.4,0.6,0.7,0.8,0.9,0.99],
                      kernel= ['linear'])
    svm_features, selector, X_train_selected, X_test_selected = feature_selection_svc(
        param_grid_SVC, False, X, X_train, y_train, X_test, y_test, n_features, verbose=0)

    ############################################################################################

    X, y = prepare_dataset("liu", dataset, categorical_features,
                           "", continous_to_binary_liu, outcome)

    X_svm = X[svm_features]
    X_new = pd.concat([X_svm, X[clinical_features]], axis=1)
    logging.info(svm_features)
    
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, random_state=split_state, test_size=0.2, shuffle=True)

    new_dir = directory + "/" + "split_random_seed_"+ str(split_state)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        
    log = open(new_dir+"/"+"myprog.log", "a+")
    sys.stdout = log
        
    grid_search_and_val(new_dir, X_train,y_train, X_test, y_test,'continuously', grid, save_model, 
                        batch_size, earlystopping_patience, n_features, epochs=1000)


# # Main

# In[3]:


class StoreDictKeyPair(argparse.Action):
    """
    Class to read in dictionary via ArgumentParser
    argparse.Action 
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def main():
    # Define args
    parser = argparse.ArgumentParser(
        description='neural network based prediction')

    subparser = parser.add_subparsers(dest='command')
    ann = subparser.add_parser('ann')
    prediction = subparser.add_parser('predict')

    # predict group
    prediction.add_argument('-j', '--json', type=str,
                            required=True, help='path to json file')
    prediction.add_argument('-w', '--weights', type=str,
                            required=True, help='path to h5 file (weights)')
    prediction.add_argument('-b', '--binary', action='store_true',
                            help='define if continuously outcome(default) or binary')
    prediction.add_argument('-t', '--table', type=argparse.FileType('r', encoding='UTF-8'),
                            required=True, help='table for predictions')

    # ann group
    ann.add_argument('-a', '--ann', type=str, required=True,
                     choices=["ann", "fs_ann", "autoencoder"], help='define which NN should be run')
    ann.add_argument('-i', '--input_data', type=argparse.FileType('r', encoding='UTF-8'),
                     required=True, help='path to input data file')
    ann.add_argument('-d','--output_dir', type=str, help='name of output directory', default="output")
    ann.add_argument('-o', '--outcome',  required=True,
                     type=str, help='define outcome variable')
    ann.add_argument('-b', '--binary', action='store_true',
                     help='define if continuously outcome (default) or binary (if binary --cutoff required)')
    ann.add_argument('-c', '--cutoff', type=int,
                     help='define cutoff for classification')
    ann.add_argument('--categorical_features', nargs='*', type=str,
                     help='list of categorical features')
    ann.add_argument('--clinical_features', nargs='*', type=str,
                     help='list of clinical features')
    ann.add_argument('-u', '--smote_upsample', action='store_true',
                     help='optional if to use SMOTE-NC (default False)')
    ann.add_argument("-g", "--grid", dest="grid", required=True, action=StoreDictKeyPair, nargs="+",
                     help='gridsearch parameters')
    ann.add_argument('-s', '--split_state', type=int, default=0,
                     help='different 80/20 split state')
    ann.add_argument('-m', '--save_model', action='store_true',
                     help='save best model')
    ann.add_argument('--n_features',type=int,default=50, 
                     help="n_features to choose for Feature Selection")
    ann.add_argument('--batch_size', type=int, default=32,
                    help="batch size for ANN")
    ann.add_argument('--early_stopping', type=int, default=20,
                    help="early stopping patience for ANN")
    args = parser.parse_args()

    if args.command == 'predict':
        load_model_and_predict(
            args.json, args.weights, args.binary, args.table)
    elif args.command == 'ann':
        # Create new directory
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d_%m_%y_%H_%M")
        time.sleep(1)
        if not os.path.exists("../results"):
            os.makedirs("../results")
        if not os.path.exists("../results/"+args.output_dir):
            os.makedirs("../results/"+args.output_dir)
        if args.binary == True:
            directory = "../results/"+args.output_dir+"/"+timestampStr+"_"+args.ann+"_binary"
        else:
            directory = "../results/"+args.output_dir+"/"+timestampStr+"_"+args.ann+"_continuously"
        time.sleep(2)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Create Logger
        logging.basicConfig(level=logging.DEBUG, filename=directory+"/logfile", filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        logging.info('Started')
        logging.info("Input File: {}".format(args.input_data))

        # if binary then cutoff required
        if args.binary and (args.cutoff is None):
            parser.error("--binary requires --cutoff")

        if args.categorical_features == None:
            categorical_features = []
        else:
            categorical_features = args.categorical_features
        if args.clinical_features == None:
            clinical_features = []
        else:
            clinical_features = args.clinical_features

        # read in params for param grid and cast into ints/floats
        dataset = load_data(args.input_data)
        for key, value in args.grid.items():
            values = value.split(",")
            numbers = ""
            if key == "__ann_n_neurons":
                ann_neurons = [int(i) for i in values]
            elif key == "__ann_num_hidden":
                ann_num_hidden = [int(i) for i in values]
            elif key == "__ann_l1_reg":
                ann_l1_reg = [float(i) for i in values]
            elif key == "__ann_l2_reg":
                ann_l2_reg = [float(i) for i in values]
            elif key == "__ann_learning_rate":
                ann_learning_rate = [float(i) for i in values]
            elif key == "__ann_used_optimizer":
                ann_used_optimizer = [str(i) for i in values]
            elif key == "__ann_dropout_rate":
                ann_dropout_rate = [float(i) for i in values]

        # create param_grid for GridSearchCV
        param_grid = dict(ann__n_neurons=ann_neurons, ann__num_hidden=ann_num_hidden, ann__used_optimizer=ann_used_optimizer, 
                          ann__l1_reg=ann_l1_reg, ann__l2_reg=ann_l2_reg, ann__learning_rate=ann_learning_rate,
                          ann__dropout_rate=ann_dropout_rate)

        #Warnings tensorflow and matplotlib silent
        tf.get_logger().setLevel(logging.ERROR)
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
        
        # ann with or without feature selection
        if args.ann == "ann":  # without feature selection
            logging.info("-----Runs neural network-----")
            print("-----Runs neural network-----")
            if args.binary == True:  # binary
                logging.info('-----Classification-----')
                print("-----Classification-----")
                logging.info("Outcome: {}".format(args.outcome))
                logging.info("Cutoff: {}".format(args.cutoff))
                logging.info("Categorical Features: {}".format(
                    args.categorical_features))
                logging.info("Clinical Features: {}".format(
                    args.clinical_features))
                logging.info("Smote: {}".format(args.smote_upsample))
                logging.info("Chosen Grid: {}".format(args.grid))
                logging.info("80/20 Split State: {}".format(args.split_state))
                logging.info("Early stopping patience: {}".format(args.early_stopping))
                logging.info("Batch size ANN: {}".format(args.batch_size))
                ann_binary(directory, dataset, args.outcome, args.cutoff, categorical_features,
                           args.smote_upsample, param_grid, "binary", args.split_state, args.save_model,
                           args.batch_size,args.early_stopping)
            else:  # continuously
                print("-----Regression-----")
                logging.info("-----Regression-----")
                logging.info("Outcome: {}".format(args.outcome))
                logging.info("Categorical Features: {}".format(
                    args.categorical_features))
                logging.info("Clinical Features: {}".format(
                    args.clinical_features))
                logging.info("Smote: {}".format(args.smote_upsample))
                logging.info("Chosen Grid: {}".format(args.grid))
                logging.info("80/20 Split State: {}".format(args.split_state))
                logging.info("Early stopping patience: {}".format(args.early_stopping))
                logging.info("Batch size ANN: {}".format(args.batch_size))
                ann_conti(directory, dataset, args.outcome, args.cutoff, categorical_features,
                          args.smote_upsample, param_grid, args.split_state, args.save_model,
                         args.batch_size, args.early_stopping)

        elif args.ann == "fs_ann":  # with feature selection
            print("-----Runs neural network with feature selection-----")
            logging.info("Runs neural network with feature selection")
            if args.binary == True:  # binary
                print("-----Classification-----")
                logging.info("-----Classification-----")
                logging.info("Outcome: {}".format(args.outcome))
                logging.info("Cutoff: {}".format(args.cutoff))
                logging.info("Categorical Features: {}".format(
                    args.categorical_features))
                logging.info("Clinical Features: {}".format(
                    args.clinical_features))
                logging.info("Smote: {}".format(args.smote_upsample))
                logging.info("Chosen Grid: {}".format(args.grid))
                logging.info("80/20 Split State: {}".format(args.split_state))
                logging.info("N_features chosen: {}".format(args.n_features))
                logging.info("Early stopping patience: {}".format(args.early_stopping))
                logging.info("Batch size ANN: {}".format(args.batch_size))
                ann_fs_binary(directory, dataset, args.outcome, args.cutoff, categorical_features,
                              clinical_features, args.smote_upsample, param_grid, args.split_state, args.save_model,
                              args.n_features,args.batch_size, args.early_stopping)

            else:  # continuously
                print("-----Regression-----")
                logging.info("-----Regression-----")
                logging.info("Outcome: {}".format(args.outcome))
                logging.info("Categorical Features: {}".format(
                    args.categorical_features))
                logging.info("Clinical Features: {}".format(
                    args.clinical_features))
                logging.info("Smote: {}".format(args.smote_upsample))
                logging.info("Chosen Grid: {}".format(args.grid))
                logging.info("80/20 Split State: {}".format(args.split_state))
                logging.info("N_features chosen: {}".format(args.n_features))
                logging.info("Early stopping patience: {}".format(args.early_stopping))
                logging.info("Batch size ANN: {}".format(args.batch_size))
                ann_fs_conti(directory, dataset, args.outcome, args.cutoff, categorical_features,
                             clinical_features, args.smote_upsample, param_grid, args.split_state, args.save_model,
                             args.n_features, args.batch_size, args.early_stopping)

        elif args.ann == "autoencoder":
            print("-----Runs neural network with autoencoder-----")
            logging.info("-----Runs neural network with autoencoder-----")
            if args.binary == True:  # binary
                print("-----Classification-----")
                logging.info("Outcome: {}".format(args.outcome))
                logging.info("Cutoff: {}".format(args.cutoff))
                logging.info("Categorical Features: {}".format(
                    args.categorical_features))
                logging.info("Clinical Features: {}".format(
                    args.clinical_features))
                logging.info("Smote: {}".format(args.smote_upsample))
                logging.info("Chosen Grid: {}".format(args.grid))
                logging.info("80/20 Split State: {}".format(args.split_state))
                logging.info("Early stopping patience: {}".format(args.early_stopping))
                logging.info("Batch size ANN: {}".format(args.batch_size))
                ann_binary(directory, dataset, args.outcome, args.cutoff, categorical_features,
                           args.smote_upsample, param_grid, "autoencoder", args.split_state, args.save_model,
                          args.batch_size, args.early_stopping)

        logging.info('Finished')


if __name__ == "__main__":
    main()

