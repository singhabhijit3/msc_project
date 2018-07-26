# coding: utf-8

# In[1]:


from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import CuDNNLSTM, multiply, add
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical, multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import argparse
import pdb


# In[2]:


data_path = "/home/s1788323/msc_project"


# In[3]:


"""parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path"""


# In[4]:


run_opt = 1


# In[5]:


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()
    
    # have removed .decode("utf-8") from return f.read().decode("utf-8").replace("\n", "<eos>").split()


# In[6]:


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


# In[7]:


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# In[8]:


def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "ptb_train.txt")
    valid_path = os.path.join(data_path, "ptb_val.txt")
    test_path = os.path.join(data_path, "ptb_test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    #print(train_data[:5])
    #print(word_to_id)
    print(vocabulary)
    #print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()


# In[9]:


class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y


# In[10]:


num_steps = 35
batch_size = 20
n_experts = 10
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)


# In[11]:


hidden_size = 650
use_dropout=True

inp = Input(shape=(num_steps,), dtype='int32')
embed = Embedding(vocabulary, hidden_size, input_length=num_steps)(inp)
if use_dropout:
    d1 = Dropout(0.5)(embed)
l1 = CuDNNLSTM(hidden_size, return_sequences=True)(d1)
if use_dropout:
    d2 = Dropout(0.5)(l1)
l2 = CuDNNLSTM(hidden_size, return_sequences=True)(d2)
if use_dropout:
    d3 = Dropout(0.5)(l2)
    
latent = TimeDistributed(Dense(n_experts*hidden_size, activation='tanh'))(d3)
latent_reshape = Reshape((-1,hidden_size))(latent)

prior = TimeDistributed(Dense(n_experts, use_bias=False, activation='softmax'))(d3)

prior = Reshape((-1,n_experts,1))(prior)

prob = TimeDistributed(Dense(vocabulary, activation='softmax'))(latent_reshape)
prob = Reshape((-1,n_experts,vocabulary))(prob)
prob = multiply([prob, prior])
prob = Lambda(lambda x: K.sum(x, axis=2))(prob)

#prob = Lambda(lambda x: x+1e-8)(prob)
model_output = prob

with tf.device("/cpu:0"):
    lstm_model = Model(inputs=inp, outputs=model_output)
    
lstm_model = multi_gpu_model(lstm_model, gpus=4)


# In[12]:


optim = SGD(lr=1)
lstm_model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['categorical_accuracy'])


# In[13]:


print(lstm_model.summary())
#checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=1, verbose=1)
num_epochs = 50
if run_opt == 1:
    lstm_model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[earlystopping, reduce_lr])#, callbacks=[checkpointer])
    # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
    #                     validation_data=valid_data_generator.generate(),
    #                     validation_steps=10)
    #model.save(data_path + "final_model.hdf5")
elif run_opt == 2:
    #model = load_model(data_path + "\model-40.hdf5")
    model = load_model(data_path + "final_model.hdf5")
    dummy_iters = 40
    example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary,
                                                     skip_step=1)
    print("Training data:")
    for i in range(dummy_iters):
        dummy = next(example_training_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps-1, :])
        true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
    # test data set
    dummy_iters = 40
    example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary,
                                                     skip_step=1)
    print("Test data:")
    for i in range(dummy_iters):
        dummy = next(example_test_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_test_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps - 1, :])
        true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
