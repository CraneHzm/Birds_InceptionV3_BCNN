# Copyright (c) 2018/4/12 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.
# Create the train, validation & test dataset from the given train, validation & test images.


# import the libs(libs & future libs).
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os  
import tensorflow as tf  
# Image is used to process images.
from PIL import Image 


# Get the current work directory.
cwd = os.getcwd()


# a class to write records to a TFRecords file.
validation_writer = tf.python_io.TFRecordWriter("validation.tfrecords") 
test_writer = tf.python_io.TFRecordWriter("test.tfrecords") 
# the augmented train set.
train_aug_writer = tf.python_io.TFRecordWriter("train_aug.tfrecords")


# the width & height to resize the images.
# (299, 299) is the image size required by Inception V3.
ImageWidth = 299
ImageHeight= 299


# create the validation tfrecords.
# read the whole 200 classes and write them into validation tfrecords.
for i in range(1, 201):
	# create a list of the files in the given class directory. 
	class_dir = os.listdir(cwd + "/Validation/" + str(i) + "/")
	for name in class_dir:
		# the path of an image.
		image_path = cwd + "/Validation/" + str(i) + "/" + name
		# print(image_path)
		# read the image.		
		img = Image.open(image_path)  
		# print(image_path)
		# print(img.mode)
		# resize the image.
		img = img.resize((ImageWidth, ImageHeight))  
		# convert the image to bytes(raw data).
		img_raw = img.tobytes()
		# calculate the label of this image.
		label = i-1
		# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
		# example defines the data format.
		example = tf.train.Example(features=tf.train.Features(feature={  
		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
		'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
		}))  
	
		# Serialize the data to string and write it.
		validation_writer.write(example.SerializeToString())
		print('Image Name: {}, label: {}'.format(name, label))  

# close the TFRecords writer.	
validation_writer.close() 


# create the test tfrecords.
# read the whole 200 classes and write them into test tfrecords.
for i in range(1, 201):
	# create a list of the files in the given class directory. 
	class_dir = os.listdir(cwd + "/Test/" + str(i) + "/")
	for name in class_dir:
		# the path of an image.
		image_path = cwd + "/Test/" + str(i) + "/" + name
		# print(image_path)
		# read the image.		
		img = Image.open(image_path)  
		# resize the image.
		img = img.resize((ImageWidth, ImageHeight))  
		# convert the image to bytes(raw data).
		img_raw = img.tobytes()
		# calculate the label of this image.
		label = i-1
		# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
		# example defines the data format.
		example = tf.train.Example(features=tf.train.Features(feature={  
		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
		'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
		}))  
	
		# Serialize the data to string and write it.
		test_writer.write(example.SerializeToString())
		print('Image Name: {}, label: {}'.format(name, label))  

# close the TFRecords writer.	
test_writer.close()


# Create the augmented train dataset, i.e., write the train data, trainFlip data, trainCrop data & trainNoise data into the dataset.

# Write the train data into train_aug tfrecords.
# read the whole 200 classes and write them into train_aug tfrecords.
for i in range(1, 201):
	# create a list of the files in the given class directory. 
	class_dir = os.listdir(cwd + "/Train/" + str(i) + "/")
	for name in class_dir:
		# the path of an image.
		image_path = cwd + "/Train/" + str(i) + "/" + name
		# print(image_path)
		# read the image.		
		img = Image.open(image_path)  
		# resize the image.
		img = img.resize((ImageWidth, ImageHeight))  
		# convert the image to bytes(raw data).
		img_raw = img.tobytes()
		# calculate the label of this image.
		label = i-1
		# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
		# example defines the data format.
		example = tf.train.Example(features=tf.train.Features(feature={  
		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
		'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
		}))  
	
		# Serialize the data to string and write it.
		train_aug_writer.write(example.SerializeToString())
		print('Image Name: {}, label: {}'.format(name, label))  

# Write the trainFlip data into train_aug tfrecords.
# read the whole 200 classes and write them into train_aug tfrecords.
for i in range(1, 201):
	# create a list of the files in the given class directory. 
	class_dir = os.listdir(cwd + "/TrainFlip/" + str(i) + "/")
	for name in class_dir:
		# the path of an image.
		image_path = cwd + "/TrainFlip/" + str(i) + "/" + name
		# print(image_path)
		# read the image.		
		img = Image.open(image_path)  
		# resize the image.
		img = img.resize((ImageWidth, ImageHeight))  
		# convert the image to bytes(raw data).
		img_raw = img.tobytes()
		# calculate the label of this image.
		label = i-1
		# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
		# example defines the data format.
		example = tf.train.Example(features=tf.train.Features(feature={  
		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
		'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
		}))  
	
		# Serialize the data to string and write it.
		train_aug_writer.write(example.SerializeToString())
		print('Image Name: {}, label: {}'.format(name, label))  		
		
# Write the trainCrop data into train_aug tfrecords.
# read the whole 200 classes and write them into train_aug tfrecords.
for i in range(1, 201):
	# create a list of the files in the given class directory. 
	class_dir = os.listdir(cwd + "/TrainCrop/" + str(i) + "/")
	for name in class_dir:
		# the path of an image.
		image_path = cwd + "/TrainCrop/" + str(i) + "/" + name
		# print(image_path)
		# read the image.		
		img = Image.open(image_path)  
		# resize the image.
		img = img.resize((ImageWidth, ImageHeight))  
		# convert the image to bytes(raw data).
		img_raw = img.tobytes()
		# calculate the label of this image.
		label = i-1
		# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
		# example defines the data format.
		example = tf.train.Example(features=tf.train.Features(feature={  
		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
		'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
		}))  
	
		# Serialize the data to string and write it.
		train_aug_writer.write(example.SerializeToString())
		print('Image Name: {}, label: {}'.format(name, label)) 		
		
# Write the trainNoise data into train_aug tfrecords.
# read the whole 200 classes and write them into train_aug tfrecords.
for i in range(1, 201):
	# create a list of the files in the given class directory. 
	class_dir = os.listdir(cwd + "/TrainNoise/" + str(i) + "/")
	for name in class_dir:
		# the path of an image.
		image_path = cwd + "/TrainNoise/" + str(i) + "/" + name
		# print(image_path)
		# read the image.		
		img = Image.open(image_path)  
		# resize the image.
		img = img.resize((ImageWidth, ImageHeight))  
		# convert the image to bytes(raw data).
		img_raw = img.tobytes()
		# calculate the label of this image.
		label = i-1
		# print('num: {}, Number: {}, label: {}'.format(num, Number, label))
		# example defines the data format.
		example = tf.train.Example(features=tf.train.Features(feature={  
		"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),  
		'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
		}))  
	
		# Serialize the data to string and write it.
		train_aug_writer.write(example.SerializeToString())
		print('Image Name: {}, label: {}'.format(name, label)) 			

# close the TFRecords writer.	
train_aug_writer.close()

