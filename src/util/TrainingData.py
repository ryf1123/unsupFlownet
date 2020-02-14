import tensorflow as tf
from components import *
import data_input

# yufan

import numpy as np
import os

class FlyingObjectInput():
    def __init__(self,batchSize,instanceParams,shuffle=True):
		self.batch_size = batchSize
		borderThicknessH = instanceParams["borderThicknessH"]
		borderThicknessW = instanceParams["borderThicknessW"]

		# from termcolor import colored
		# print(colored("changing image size to 64", "green"))
		# borderThicknessH, borderThicknessW = 64, 64
		# assert borderThicknessH == 64
		# assert borderThicknessW == 64
        
		self.data = self.load_data()
		batch = self.first1k_train()
		## queuing complete
		# import ipdb; ipdb.set_trace()
		# mean subtraction
		mean = [[[[0.448553, 0.431021, 0.410602]]]]

		# print(img0raw)
		img0raw = tf.cast(batch[0],tf.float32)/255.0 - mean
		img1raw = tf.cast(batch[1],tf.float32)/255.0 - mean

		## async section done ##

		#image augmentation
		photoParam = photoAugParam(batchSize,0.7,1.3,0.2,0.9,1.1,0.7,1.5,0.00)
		imData0aug = photoAug(img0raw,photoParam) - mean
		imData1aug = photoAug(img1raw,photoParam) - mean

		# artificial border augmentation
		borderMask = validPixelMask(tf.stack([1, \
			img0raw.get_shape()[1], \
			img0raw.get_shape()[2], \
			1]),borderThicknessH,borderThicknessW)

		imData0aug *= borderMask
		imData1aug *= borderMask

		#LRN skipped
		lrn0 = tf.nn.local_response_normalization(img0raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)
		lrn1 = tf.nn.local_response_normalization(img1raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)

		#gradient images
		imData0Gray = rgbToGray(img0raw)
		imData1Gray = rgbToGray(img1raw)

		imData0Grad = gradientFromGray(imData0Gray)
		imData1Grad = gradientFromGray(imData1Gray)

		# ----------expose tensors-----------

		self.frame0 = {
			"rgb": imData0aug,
			"rgbNorm": lrn0,
			"grad": imData0Grad
		}

		self.frame1 = {
			"rgb": imData1aug,
			"rgbNorm": lrn1,
			"grad": imData1Grad
		}

		self.validMask = borderMask

    
    def load_data(self):
        from termcolor import colored
        print(colored("offset dropped", "red"))
        import h5py
        self.data_path = "../data/flying"
        hf = h5py.File(os.path.join(self.data_path, 'syn_final_train.h5'), 'r')

        # print("Warning!!! using data unseen")
        # im1 = self._resize_image(np.array(hf.get('image_before'))[1000:]) # / 255.0 * 2.0 - 1
        # im2 = self._resize_image(np.array(hf.get('image_after'))[1000:]) # / 0255.0 * 2.0 - 1
        # flow = self._resize_flow(np.array(hf.get('flow'))[1000:])

        # im1 = self._resize_image(np.array(hf.get('image_before'))[:1000]) # / 255.0 * 2.0 - 1
        # im2 = self._resize_image(np.array(hf.get('image_after'))[:1000]) # / 0255.0 * 2.0 - 1
        # flow = self._resize_flow(np.array(hf.get('flow'))[:1000])
        
        im1 = np.array(hf.get('image_before')[:1000]) # / 255.0 * 2.0 - 1
        im2 = np.array(hf.get('image_after')[:1000]) # / 0255.0 * 2.0 - 1
        flow = np.array(hf.get('flow')[:1000])

        from termcolor import colored
        print(colored("data len should be 1000", "red"))
        
        im1 = im1.astype("float32")
        im2 = im2.astype("float32")
        flow = flow.astype("float32")
        
        hf.close()
        return [im1, im2, flow]
    
    # def _resize_image(self, images):
    #     height, width = self.dims
    #     images_resize = np.zeros([images.shape[0], height, width, 3], dtype=np.uint8)
    #     for i in range(len(images)):
    #         #  import ipdb; ipdb.set_trace()
    #         # yufan width first for cv2
    #         images_resize[i] = cv2.resize(images[i], (width, height))
    #     return images_resize
    
    # def _resize_flow(self, flows):
    #     height, width = self.dims
    #     flows_resize = np.zeros([flows.shape[0], height, width, 2], dtype=float)
    #     for i in range(len(flows)):
    #         flows_resize[i] = resampleFlow(flows[i], [height, width])
    #     return flows_resize
    
    def first1k_train(self, offset=None):
        data = self.data
        assert data!=None
        
        im1, im2, _ = data

        im1 = tf.train.slice_input_producer([im1],shuffle=False)
        im2 = tf.train.slice_input_producer([im2],shuffle=False)
        
        im1, im2 = im1[0], im2[0]
        
        return tf.train.batch(
            [im1, im2],
            batch_size=self.batch_size,
            # num_threads=self.num_threads,
            allow_smaller_final_batch=False)
    
#     def first1k_test(self, offset=None):
#         data = self.data
#         assert data!=None
        
#         im1, im2, flow = data
#         input_shape = im1.shape
# #         import ipdb; ipdb.set_trace()
#         mask = np.zeros(im1.shape[:-1]+(1,)).astype("float32")
# #         import ipdb; ipdb.set_trace()
#         im1 = tf.train.slice_input_producer([im1],shuffle=False, num_epochs=1)
#         im2 = tf.train.slice_input_producer([im2],shuffle=False, num_epochs=1)
#         flow = tf.train.slice_input_producer([flow],shuffle=False, num_epochs=1)
#         mask = tf.train.slice_input_producer([mask],shuffle=False, num_epochs=1) # ,num_epochs=1
        
#         im1, im2, flow, mask = im1[0], im2[0], flow[0], mask[0]
#         input_shape = tf.shape(im1)
# #         import ipdb; ipdb.set_trace()
        
#         return tf.train.batch(
#             [im1, im2, input_shape, flow, mask],
#             batch_size=self.batch_size,
#             num_threads=self.num_threads,
#             allow_smaller_final_batch=False)    
    
class TrainingData:
	"""
	handles queuing and preprocessing prior to and after batching
	"""
	def __init__(self,batchSize,instanceParams,shuffle=True):
		with tf.variable_scope(None,default_name="ImagePairData"):
			borderThicknessH = instanceParams["borderThicknessH"]
			borderThicknessW = instanceParams["borderThicknessW"]
			if instanceParams["dataset"] == "kitti2012" or instanceParams["dataset"] == "kitti2015":
				datasetRoot = "../example_data/"
				frame0Path = datasetRoot+"datalists/train_im0.txt"
				frame1Path = datasetRoot+"datalists/train_im1.txt"
				desiredHeight = 320
				desiredWidth = 1152
			elif instanceParams["dataset"] == "sintel":
				datasetRoot = "/home/jjyu/datasets/Sintel/"
				frame0Path = datasetRoot+"datalists/train_raw_im0.txt"
				frame1Path = datasetRoot+"datalists/train_raw_im1.txt"
				desiredHeight = 384
				desiredWidth = 960
			elif instanceParams["dataset"] == "flyingobjects":
				datasetRoot = "/home/jjyu/datasets/Sintel/"
				frame0Path = datasetRoot+"datalists/train_raw_im0.txt"
				frame1Path = datasetRoot+"datalists/train_raw_im1.txt"
				desiredHeight = 64
				desiredWidth = 64				
			else:
				assert False, "unknown dataset: " + instanceParams["dataset"]


			# create data readers
			frame0Reader = data_input.reader.Png(datasetRoot,frame0Path,3)
			frame1Reader = data_input.reader.Png(datasetRoot,frame1Path,3)

			#create croppers since kitti images are not all the same size
			cropShape = [desiredHeight,desiredWidth]
			cropper = data_input.pre_processor.SharedCrop(cropShape,frame0Reader.data_out)

			dataReaders = [frame0Reader,frame1Reader]
			DataPreProcessors = [[cropper],[cropper]]

			self.dataQueuer = data_input.DataQueuer(dataReaders,DataPreProcessors,n_threads=batchSize*4)

			# place data into batches, order of batches matches order of datareaders
			batch = self.dataQueuer.queue.dequeue_many(batchSize)

			## queuing complete
			import ipdb; ipdb.set_trace() 

			# mean subtraction
			mean = [[[[0.448553, 0.431021, 0.410602]]]]
			img0raw = tf.cast(batch[0],tf.float32)/255.0 - mean
			img1raw = tf.cast(batch[1],tf.float32)/255.0 - mean

			## async section done ##

			#image augmentation
			photoParam = photoAugParam(batchSize,0.7,1.3,0.2,0.9,1.1,0.7,1.5,0.00)
			imData0aug = photoAug(img0raw,photoParam) - mean
			imData1aug = photoAug(img1raw,photoParam) - mean

			# artificial border augmentation
			borderMask = validPixelMask(tf.stack([1, \
				img0raw.get_shape()[1], \
				img0raw.get_shape()[2], \
				1]),borderThicknessH,borderThicknessW)

			imData0aug *= borderMask
			imData1aug *= borderMask

			#LRN skipped
			lrn0 = tf.nn.local_response_normalization(img0raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)
			lrn1 = tf.nn.local_response_normalization(img1raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)

			#gradient images
			imData0Gray = rgbToGray(img0raw)
			imData1Gray = rgbToGray(img1raw)

			imData0Grad = gradientFromGray(imData0Gray)
			imData1Grad = gradientFromGray(imData1Gray)

			# ----------expose tensors-----------

			self.frame0 = {
				"rgb": imData0aug,
				"rgbNorm": lrn0,
				"grad": imData0Grad
			}

			self.frame1 = {
				"rgb": imData1aug,
				"rgbNorm": lrn1,
				"grad": imData1Grad
			}

			self.validMask = borderMask
