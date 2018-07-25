import keras
import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_squared_error
from skimage import measure
import math

import keras.backend as K

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Save_result(keras.callbacks.Callback):
    
    def __init__(self,save_file_name, Y, img_x, img_y):
       
        self.save_file_name = save_file_name
        self.test_images = Y[:,:,:,:2]
        
        self.num_of_test_image = self.test_images.shape[0]
        self.x_image = img_x
        self.y_image = img_y
        
        self.X_tedata = np.zeros((self.num_of_test_image,self.x_image,self.y_image,1))
        self.X_tedata[:,:,:,0] = self.test_images[:,:,:,1]

        self.X_tedata -= 0.5
        self.X_tedata /= 0.2
        
        self.i = 0
        
        self.PSNR_denoised_image_arr = []
        self.SSIM_denoised_image_arr = []
        
        self.PSNR_arr = []
        self.SSIM_arr = []

        self.sup_PSNR_arr = []
        self.sup_SSIM_arr = []
        
        self.training_loss_arr = []
        
        
    def get_PSNR(self, X, X_hat):
        
        mse = mean_squared_error(X,X_hat)
        test_PSNR = 10 * math.log10(1/mse)
        
        return test_PSNR
    
    def get_SSIM(self, X, X_hat):
        
        test_SSIM = measure.compare_ssim(X, X_hat, dynamic_range=X.max() - X.min())
        
        return test_SSIM
        
    def generate_flipped_image_set(self, X_data):
        
        if X_data.shape[2] == 1:
        
            flipped_image_set = []

            lr_flip = np.fliplr(X_data.reshape(self.x_image,self.y_image))
            ud_flip = np.flipud(X_data.reshape(self.x_image,self.y_image))
            lr_ud_flip = np.flipud(lr_flip)

            flipped_image_set = X_data.reshape(1,self.x_image,self.y_image,X_data.shape[2])
            flipped_image_set = np.vstack((flipped_image_set, lr_flip.reshape(1,self.x_image,self.y_image,X_data.shape[2])))
            flipped_image_set = np.vstack((flipped_image_set, ud_flip.reshape(1,self.x_image,self.y_image,X_data.shape[2])))
            flipped_image_set = np.vstack((flipped_image_set, lr_ud_flip.reshape(1,self.x_image,self.y_image,X_data.shape[2])))
            
        else:
            
            flipped_image_set = np.zeros((4,X_data.shape[0],X_data.shape[1],X_data.shape[2]))
            
            for i in range(3):
                
                origin = X_data[:,:,i]
                lr_flip = np.fliplr(X_data[:,:,i])
                ud_flip = np.flipud(X_data[:,:,i])
                lr_ud_flip = np.flipud(X_data[:,:,i])

                flipped_image_set[0,:,:,i] = origin
                flipped_image_set[1,:,:,i] = lr_flip
                flipped_image_set[2,:,:,i] = ud_flip
                flipped_image_set[3,:,:,i] = lr_ud_flip

        return flipped_image_set
    
    def reverse_flipped_image_set(self,X_data):
        
        origin_image = X_data[0]
        reverse_lr_flip = np.fliplr(X_data[1])
        reverse_ud_flip = np.flipud(X_data[2])
        reverse_lr_ud_flip = np.flipud(np.fliplr(X_data[3]))
        
        ensemble_image = (origin_image + reverse_lr_flip + reverse_ud_flip + reverse_lr_ud_flip)/4
    
        return ensemble_image
        
    def test_model(self,epoch):
        
        self.X_tedata_flip = self.X_tedata[self.i,:,:,:]
        self.X_tedata_flip = self.generate_flipped_image_set(self.X_tedata_flip)
        
        self.Z_tedata_flip = self.test_images[self.i,:,:,1].reshape(self.x_image,self.y_image,1)
        self.Z_tedata_flip = self.generate_flipped_image_set(self.Z_tedata_flip)
        
        returned_score = self.model.predict(self.X_tedata_flip.reshape(4,self.x_image,self.y_image,1),batch_size=4, verbose=0)
        returned_score = np.array(returned_score)
        returned_score = returned_score.reshape(4,self.x_image,self.y_image,2)

        denoised_big_test_image = returned_score[:,:,:,0] * (self.Z_tedata_flip[:,:,:,0]) + returned_score[:,:,:,1]
        denoised_big_test_image = np.clip(denoised_big_test_image, 0, 1)

        denoised_ensemble_test_image = self.reverse_flipped_image_set(denoised_big_test_image)
        
        PSNR = self.get_PSNR(self.test_images[self.i,:,:,0],denoised_ensemble_test_image)
        SSIM = self.get_SSIM(self.test_images[self.i,:,:,0],denoised_ensemble_test_image)
        
        self.temp_PSNR_arr.append(PSNR)
        self.temp_SSIM_arr.append(SSIM)

        if self.best_PSNR[1] < PSNR:
            self.best_PSNR[0] = epoch
            self.best_PSNR[1] = PSNR
            self.temp_PSNR_denoised_image_arr = denoised_ensemble_test_image 
            
        if self.best_SSIM[1] < SSIM:
            self.best_SSIM[0] = epoch
            self.best_SSIM[1] = SSIM
            self.temp_SSIM_denoised_image_arr = denoised_ensemble_test_image 

        return PSNR, SSIM
    
    def test_model_zero(self):
        
        self.X_tedata_flip = self.X_tedata[self.i,:,:,:]
        self.X_tedata_flip = self.generate_flipped_image_set(self.X_tedata_flip)
        
        self.Z_tedata_flip = self.test_images[self.i,:,:,1].reshape(self.x_image,self.y_image,1)
        self.Z_tedata_flip = self.generate_flipped_image_set(self.Z_tedata_flip)
        
        returned_score = self.model.predict(self.X_tedata_flip.reshape(4,self.x_image,self.y_image,1),batch_size=1, verbose=0)
        returned_score = np.array(returned_score)
        returned_score = returned_score.reshape(4,self.x_image,self.y_image,2)

        denoised_big_test_image = returned_score[:,:,:,0] * (self.Z_tedata_flip[:,:,:,0]) + returned_score[:,:,:,1]
        denoised_big_test_image = np.clip(denoised_big_test_image, 0, 1)

        PSNR = self.get_PSNR(self.test_images[self.i,:,:,0],denoised_big_test_image[0])
        SSIM = self.get_SSIM(self.test_images[self.i,:,:,0],denoised_big_test_image[0])

        self.sup_PSNR_arr.append(PSNR)
        self.sup_SSIM_arr.append(SSIM)
        
        print (PSNR, SSIM)

        return 
    
    def on_train_begin(self, logs={}):
        self.test_model_zero()
        self.temp_PSNR_arr = []
        self.temp_SSIM_arr = []
        self.best_PSNR = np.zeros((2,))
        self.best_SSIM = np.zeros((2,))
        self.temp_PSNR_denoised_image_arr = []
        self.temp_SSIM_denoised_image_arr = []
        self.temp_training_loss_arr = []
        return
        
    def on_train_end(self, logs={}):
        self.i += 1
        
        self.PSNR_denoised_image_arr.append(self.temp_PSNR_denoised_image_arr)
        self.SSIM_denoised_image_arr.append(self.temp_SSIM_denoised_image_arr)
        
        self.PSNR_arr.append(self.temp_PSNR_arr)
        self.SSIM_arr.append(self.temp_SSIM_arr)
        
        self.training_loss_arr.append(self.temp_training_loss_arr)
        
        sio.savemat('./result_data/'+self.save_file_name+'_Result' ,{'loss_arr_epoch':self.training_loss_arr,'SSIM_arr_epoch':self.SSIM_arr,'PSNR_arr_epoch':self.PSNR_arr,
                                                                    'max_SSIM_denoised_image_arr':self.SSIM_denoised_image_arr,'max_PSNR_denoised_image_arr':self.PSNR_denoised_image_arr,
                                                                   'sup_PSNR_arr':self.sup_PSNR_arr,'sup_SSIM_arr':self.sup_SSIM_arr})
        
        return
    def set_i(self,i):
        self.i = i
        
    def on_epoch_end(self, epoch, logs={}):
       
        PSNR, SSIM = self.test_model(epoch)
        
        self.temp_training_loss_arr.append(logs.get('loss'))
        
        print ('image : ' + str(self.i+1) + '-th' + ' epoch : ' + str(epoch+1))
        print ('PSNR : ' + str(PSNR) + ' Best Epoch : ' + str(self.best_PSNR[0])+ ' Best PSNR : ' + str(self.best_PSNR[1]))
        print ('SSIM : ' + str(SSIM) + ' Best Epoch : ' + str(self.best_SSIM[0])+ ' Best SSIM : ' + str(self.best_SSIM[1]))
        
        