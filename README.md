# PUBG winner prediction project

## Background
This repository contains Jupyter notebooks and python scripts for EECS 475 and 495 project at Northwestern University. This project is derived from a [Kaggle competition: PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction). The aim is to comprehensively apply and compare machine learning and optimization methods we learned in class to predict the final winning placements of players in Player's Unknown Battle Ground (PUBG).

## Prerequisites
* Python 3.6

## File desciptions
All scripts and notebooks are inside Codes/ folder. Specifically, I compared use of different losses as baselines (absolute error, least square error, softmax) in combination with 5 types of optimizers: 
* Normalized gradient descent
* Component-wise gradient descent
* Momentum-accelerated gradient descent
* Adam
* RMSProp

These baselines were compared with more advanced ML/DL algorithms, including
* Perceptron (in combination with 5 types of optimizers)
* Decision Tree
* Random Forest
* XGBoost 
* LightGBM
* DNN
* ResNet
