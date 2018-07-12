import keras
import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_squared_error
import math

import keras.backend as K

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Get_results(keras.callbacks.Callback):
    def __init__(self,save_file_name, test_file_name, model, model_for_test):
        
        self.num_of_test_image = 68
        self.save_file_name = save_file_name
        
        file_name = test_file_name
        f = sio.loadmat('./data/'+file_name)
        
        self.clean_images = np.array(f["clean_images"]) / 255.
        self.noisy_images = np.array(f["noisy_images"]) / 255.

        self.test_images = np.zeros((2,68,321,481))
        self.test_images[0] = self.clean_images
        self.test_images[1] = self.noisy_images
        
        self.X_tedata = np.zeros((68,321,481,1))
        self.X_tedata[:,:,:,0] = self.noisy_images

        self.X_tedata -= 0.5
        self.X_tedata /= 0.2
        
        self.model_for_saving = model
        self.model_for_test = model_for_test
        
        print ('Save result class is called!')
        
    def test_model(self,epoch):
        self.model_for_saving.save_weights('./weights/'+self.save_file_name+'_for_training.hdf5')
        self.model_for_test.load_weights('./weights/'+self.save_file_name+'_for_training.hdf5')
        for num_of_image in range(0,self.num_of_test_image):
            returned_score = self.model_for_test.predict(self.X_tedata[num_of_image,:,:,:].reshape(1,321,481,1),batch_size=1, verbose=0)
            returned_score = np.array(returned_score)
            returned_score = returned_score.reshape(321,481,2)

            self.denoised_big_test_image[num_of_image] = returned_score[:,:,0] * (self.test_images[1,num_of_image,:,:]) + returned_score[:,:,1]

        for i in range(0,68):
            mse = mean_squared_error(self.test_images[0,i,:,:],self.denoised_big_test_image[i,:,:])
            PSNR_ground_denoised = 10 * math.log10(1/mse)
            self.result_of_test_PSNR[i] = PSNR_ground_denoised
    #         print 'PSNR of ' + name_list_big_test_image[i] +' image : ', PSNR_ground_denoised


        mean_of_result_of_test_PSNR = np.mean(self.result_of_test_PSNR)
        self.list_of_test_PSNR.append(mean_of_result_of_test_PSNR)

        if self.best_PSNR[1] < np.mean(self.result_of_test_PSNR):
            self.best_PSNR[0] = epoch+1
            self.best_PSNR[1] = mean_of_result_of_test_PSNR
            self.list_of_best_PSNR[:] = self.result_of_test_PSNR[:]
            self.saving_denoised_big_test_image[:,:,:] = self.denoised_big_test_image[:,:,:]
            self.model_for_saving.save_weights('./weights/'+self.save_file_name+'.hdf5')

        #print mean_of_result_of_test_PSNR, self.best_PSNR[0], self.best_PSNR[1]
        return mean_of_result_of_test_PSNR, self.best_PSNR[0], self.best_PSNR[1]
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.tr_losses = []     
        self.logs = []

        self.denoised_big_test_image = np.zeros((self.num_of_test_image,321,481))
        self.result_of_test_PSNR = np.zeros((self.num_of_test_image,))
        self.list_of_test_PSNR = []
        self.list_of_best_PSNR = np.zeros((self.num_of_test_image,))
        self.best_PSNR = np.zeros((2,))

        self.saving_denoised_big_test_image = np.zeros((self.num_of_test_image,321,481))
        
    def on_epoch_end(self, epoch, logs={}):

        mean, best_ep, best_PSNR = self.test_model(epoch)
        
        print ('\n epoch : ' + str(epoch+1) +' mean_PSNR : ' + str(mean) + ' Best Epoch : ' + str(best_ep,)+ ' Best PSNR : ' + str(best_PSNR))

        self.logs.append(logs)
        self.x.append(self.i)
        self.tr_losses.append(logs.get('loss'))
        self.i += 1
        
        plt.figure(figsize=[10,8])
        plt.plot(self.x, self.tr_losses, label="loss")
        plt.legend()
        plt.savefig('./result_data/'+self.save_file_name+'_trloss.png')
        plt.clf()
        
        plt.figure(figsize=[10,8])
        plt.plot(self.x, self.list_of_test_PSNR, label="Average Test PSNR")
        plt.legend()
        plt.savefig('./result_data/'+self.save_file_name+'_tePSNR.png')
        plt.clf()

        sio.savemat('./result_data/'+self.save_file_name+'_Result',  
                    {'PSNR_arr_epoch':self.list_of_test_PSNR, 'loss_arr_epoch':self.tr_losses, 'max_PSNR_denoised_images':self.saving_denoised_big_test_image,})