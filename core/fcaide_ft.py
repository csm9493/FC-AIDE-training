from .fcaide_ft_result import Save_result
import numpy as np
import math
import scipy.io as sio


class Fine_tuning:
    
    def __init__(self, save_file_name, weight_file_name, test_file_name, sigma_arr, model):
        
        
        self.save_file_name = save_file_name
        self.weight_file_name = weight_file_name
        self.test_file_name = test_file_name
        self.sigma_arr = sigma_arr
        
        self.init_lrate = 0.0001
        self.lrate_decay = 0.0
        self.mini_batch_size = 1
        self.max_ep = 30

        
        self.PSNR_arr_epoch = []
        self.SSIM_arr_epoch = []

        self.max_PSNR = 0
        self.max_SSIM = 0

        self.max_PSNR_denoised_image_arr = []
        self.max_SSIM_denoised_image_arr = []
        
        self.make_model = model
        
        return
    
    def get_images(self):
        
        f = sio.loadmat('./data/'+self.test_file_name)

        self.noisy_images = np.float32(np.array(f["noisy_images"]))
        self.clean_images = np.float32(np.array(f["clean_images"]))

        self.num_images = self.noisy_images.shape[0]
        self.img_x = self.noisy_images.shape[1]
        self.img_y = self.noisy_images.shape[2]

        self.noisy_images /= 255.
        self.clean_images /= 255.

        self.X = np.zeros((self.num_images,self.img_x,self.img_y,1))
        self.X[:,:,:,0] = self.noisy_images

        self.X -= 0.5
        self.X /= 0.2
        
        self.Y = np.zeros((self.num_images,self.img_x,self.img_y,3))
        self.Y[:,:,:,0] = self.clean_images
        self.Y[:,:,:,1] = self.noisy_images
        for idx in range(self.num_images):
            self.Y[idx,:,:,2] = self.sigma_arr[idx] / 255.

    
    
    def generate_flipped_image_set(self, X_data):
        
        if X_data.shape[2] == 1:
        
            flipped_image_set = []

            lr_flip = np.fliplr(X_data.reshape(self.img_x,self.img_y))
            ud_flip = np.flipud(X_data.reshape(self.img_x,self.img_y))
            lr_ud_flip = np.flipud(lr_flip)

            flipped_image_set = X_data.reshape(1,self.img_x,self.img_y,X_data.shape[2])
            flipped_image_set = np.vstack((flipped_image_set, lr_flip.reshape(1,self.img_x,self.img_y,X_data.shape[2])))
            flipped_image_set = np.vstack((flipped_image_set, ud_flip.reshape(1,self.img_x,self.img_y,X_data.shape[2])))
            flipped_image_set = np.vstack((flipped_image_set, lr_ud_flip.reshape(1,self.img_x,self.img_y,X_data.shape[2])))
            
        else:
            
            flipped_image_set = np.zeros((4,X_data.shape[0],X_data.shape[1],X_data.shape[2]))
            
            for i in range(3):
                
                origin = X_data[:,:,i]
                lr_flip = np.fliplr(X_data[:,:,i])
                ud_flip = np.flipud(X_data[:,:,i])
                lr_ud_flip = np.flipud(np.fliplr(X_data[:,:,i]))

                flipped_image_set[0,:,:,i] = origin
                flipped_image_set[1,:,:,i] = lr_flip
                flipped_image_set[2,:,:,i] = ud_flip
                flipped_image_set[3,:,:,i] = lr_ud_flip

        return flipped_image_set
    
    def test_images(self):
        
        self.get_images()
        save_results = Save_result(self.save_file_name, self.Y, self.img_x, self.img_y,)
         
        for idx in range(self.num_images):
            
            X_data = self.X[idx,:]
            X_data = self.generate_flipped_image_set(X_data)
            
            y_data = self.Y[idx,:]
            y_data = self.generate_flipped_image_set(y_data)
            
            X_data = X_data.reshape(4,self.img_x,self.img_y,1)
            y_data = y_data.reshape(4,self.img_x,self.img_y,3)
            
            self.model = self.make_model()
            self.model.load_weights('./weights/' + self.weight_file_name)
            
            self.model.fit(X_data, y_data, verbose=2, batch_size = self.mini_batch_size, epochs = self.max_ep, callbacks=[save_results])
            
            del self.model
            
        