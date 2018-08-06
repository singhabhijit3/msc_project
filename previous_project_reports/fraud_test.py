'''
This test is due in 30-45min. Send this file back as you finish. 
Please use any internet resource you find suitable to answer the following questions.


1.Please make run the code and understand what it does. Solve dependency issues if they arise (10min)
  Use the dataset file to run the code properly (put in the appropriate folder to run the code)
  REMARK: Do NOT train properly just ensure that it runs with no bugs

2.Explain the code: What the code overall does?(10min) 

  Fill the following sentences replacing the '...' :
  The code below is about ...  
  and it trains a neuron network to ... 
  The neuron network is given by two parts: ... 
  
3.Use a new cell and count how many elements in the dataset are in each of the 2 classes (fraud, normal) (5min)

4. Get the paper https://arxiv.org/pdf/1802.00187.pdf (also attached in the mail) (10-15min)
   and answer the following question:
4.1 Using the formula (4) in the paper, describe how you would insert that in the code above: 
   which particular lines would you modify? Do you need a new function? What this new function will do? 
   Where this new function will be called? Help yourself with any pseudocode and any internet resource as you like.
   REMARK: do NOT spend time to read the paper details just use the bits necessary to answer the question.
   
   If you like (not necessary), fill the following sentences replacing the '...' :
   
   Formula (4) is a ...
   A new function should be written to ... which it has  input ... and output ...
   The function will apply ... to the input and then it will apply ...
   This new function is called in the code by the function ...
   
'''



import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import backend as K

%matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

#load data
df = pd.read_csv("creditcard_sample.csv")

#preprocessing data
from sklearn.preprocessing import StandardScaler

data = df.drop(['Time'], axis=1)

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values

#NN model 

input_dim = X_train.shape[1]
encoding_dim = 14


#encoder
input_tensor = Input(shape=(input_dim, ))

encoderOut = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_tensor)
encoderOut = Dense(int(encoding_dim / 2), activation="relu")(encoderOut)

encoder = Model(input_tensor, encoderOut)


#decoder
decoder_input = Input(shape=(int(encoding_dim / 2),))
decoderOut = Dense(int(encoding_dim / 2), activation='tanh',name='decoder_input')(decoder_input)
decoderOut = Dense(input_dim, activation='relu',name='decoder_output')(decoderOut)

decoder = Model(decoder_input, decoderOut)

#autoencoder
autoInput = Input(shape=(input_dim, ))
encoderOut = encoder(autoInput)
decoderOut = decoder(encoderOut)
autoencoder = Model(inputs=autoInput, outputs=decoderOut)

print input_dim

#train
nb_epoch = 3
batch_size = 32

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer]).history