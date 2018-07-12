##### from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Conv1D
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model

from keras.callbacks import TensorBoard


import h5py
import numpy as np

import scipy.io as sio
import h5py
import tensorflow as tf

import random

from .fcaide_sup_get_results import Get_results

from keras import backend as K
def custom_regression1(y_true,y_pred):    #
    return K.mean(K.square(y_true[:,:,:,0]-(y_pred[:,:,:,0]*y_true[:,:,:,1]+y_pred[:,:,:,1])))  

class Train_FCAIDE:
    
    def __init__(self, date = None ,case = None, std=25, mini_batch_size=8, trdata_flag = 0, ep=50, gpus=4):
        self.model_output = 2
        self.std = std
        self.conv_filter_size = 64
        self.gpus = gpus
        self.mini_batch_size = mini_batch_size*self.gpus
        self.trdata_flag = trdata_flag
        self.training_data_path = '../data/'
        self.epochs = ep
        if trdata_flag == 0:
            self.training_data_file_name = 'NIPS2018_384000_training_data_patch40.hdf5'
        elif trdata_flag == 1:
            self.training_data_file_name = 'NIPS2018_384000_training_data_patch50.hdf5'
        elif trdata_flag == 2:
            self.training_data_file_name = 'NIPS2018_384000_training_data_patch60.hdf5'
        else:
            self.training_data_file_name = 'NIPS2018_384000_training_data_patch50.hdf5'

        self.test_data_path = '../data/'
        self.test_data_file_name = 'NIPS2018_berkeley_test_images_std'+str(std)+'.mat'
        
        if trdata_flag == 0:
            self.save_file_name = str(date) + '_FCAIDE_384000trdata_patch40'+'_std'+str(std)
        elif trdata_flag == 1:
            self.save_file_name = str(date) + '_FCAIDE_384000trdata_patch50'+'_std'+str(std)
        elif trdata_flag == 2:
            self.save_file_name = str(date) + '_FCAIDE_384000trdata_patch60'+'_std'+str(std)
        else:
            self.save_file_name = str(date) + '_FCAIDE_384000trdata_patch50'+'_blind'
         
        print (self.save_file_name)
        if case != None :
            self.save_file_name += '_' + str(case)
            
        return
    
    def make_model(self, model = None, parallel_model=None, test_model = None, model_flag = 0):

        if model_flag == 0:
            print ('make model')

        else:
            
            self.model = model
            self.parallel_model = parallel_model
            self.test_model = test_model

        return
        
    def generate_training_sequences(self,batch_size, tr_data, sample_idxs):
        while True:
            # generate sequences for training
            x_axis = tr_data["training_images"].shape[2]
            y_axis = tr_data["training_images"].shape[3]

            training_sample_idxs = np.random.permutation(sample_idxs)
            training_sample_count = len(training_sample_idxs)
            batches = int(training_sample_count/batch_size)
            remainder_samples = training_sample_count%batch_size
            if remainder_samples:
                batches = batches + 1
            # generate batches of samples
            for idx in range(0, batches):
                if idx == batches - 1:
                    batch_idxs = training_sample_idxs[idx*batch_size:]
                else:
                    batch_idxs = training_sample_idxs[idx*batch_size:idx*batch_size+batch_size]
                batch_idxs = sorted(batch_idxs)

                batch_dataset = tr_data["training_images"][:,batch_idxs] / 255.
                
                if self.trdata_flag != 3:
                
                    noise_mean = 0
                    noise_std = self.std / 255.

                    Y = np.zeros((batch_size,x_axis,y_axis,2))
                    Y[:,:,:,0] = batch_dataset[0,:]
                    Y[:,:,:,1] = batch_dataset[0,:] + np.random.normal(noise_mean, noise_std, batch_dataset[0,:,:,:].shape)

                    X = Y[:,:,:,1]

                    X = (X - 0.5) / 0.2

                    yield X.reshape(batch_size,x_axis,y_axis,1), Y
                    
                else:
                    
                    noise_mean = 0
                
                    Y = np.zeros((batch_size,x_axis,y_axis,2))

                    Y[:,:,:,0] = batch_dataset[0,:]

                    for patch in range(len(batch_idxs)):
                        noise_std = (random.random() * 50) / 255.
                        Y[patch,:,:,1] = batch_dataset[0,patch] + np.random.normal(noise_mean, noise_std, batch_dataset[0,0,:,:].shape)

                    X = Y[:,:,:,1]

                    X = (X - 0.5) / 0.2

                    yield X.reshape(batch_size,x_axis,y_axis,1), Y
    
    def train_model(self):

        tr_data_location = './data/'+ self.training_data_file_name
        
        get_result = Get_results(self.save_file_name, self.test_data_file_name, self.model, self.test_model)
        
        tensorboard = TensorBoard(log_dir="logs/"+self.save_file_name, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
        
        with h5py.File(tr_data_location, "r") as tr_data:
            training_sample_count = tr_data["training_images"].shape[1]
            training_sample_idxs = range(0, training_sample_count)

            training_sequence_generator = self.generate_training_sequences(self.mini_batch_size, tr_data, training_sample_idxs)
            self.parallel_model.fit_generator(generator=training_sequence_generator,steps_per_epoch=(training_sample_count/(self.mini_batch_size)+1),epochs=self.epochs,callbacks=[get_result],verbose=2)

        
        






