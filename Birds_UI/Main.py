# Copyright (c) Hu Zhiming 2018/4/19 JimmyHu@pku.edu.cn All Rights Reserved.

#################### Libs. ####################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

# import class from other files.
from UI import Ui_MainWindow
from InceptionV3 import InceptionV3
from BCNN import BCNN


# inherit UI class from the Ui_MainWindow class.
class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow, InceptionV3, BCNN):
	def __init__(self):
		# Init the objects.
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		InceptionV3.__init__(self)
		BCNN.__init__(self)
		
		
		# Set up the UI.
		self.setupUi(self)
		
		# member variables.
		self.imagepath = ''
		self.classname = ''
		self.classID = -1
		self.prdclassname = ''
		self.totalCount = 0
		self.rightCount = 0
		self.wrongCount = 0
		self.accuracy = 0.0
		
		# Init our customized UI & add functions to it.
		self.initUI()
	
	def initUI(self):
		'''Init our customized UI'''
		self.SelectImage.clicked.connect(self.showImage)
		self.Predict.clicked.connect(self.imagePredict)
		self.Exit.clicked.connect(self.exitApplication)
	
	
	def exitApplication(self):
		'''Exit the application.'''
		self.close()
	
	
	def showImage(self):
		'''Select & Show an image when you click the SelectImage Button.'''
		fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')
		if fname[0]:
			
			# Clear the GT & Prd Text.
			self.GtText.setText('')
			self.PrdText.setText('')
			self.ImageLabel.setPixmap(QtGui.QPixmap(fname[0]))
			# resize the original image to fit the ImageLabel.
			self.ImageLabel.setScaledContents(True)
			# show the image.
			self.ImageLabel.show()
			# print(fname[0])
			# Get the path of the image.
			self.imagepath = fname[0]
			# print(self.imagepath)
			# parse the class name of the input image.
			self.classname = self.parseClassname(self.imagepath)
			self.classID = int(self.classname)
			self.GtText.setText(self.classname)
			# print(self.classname)
			
			
	def parseClassname(self, imagepath):
		'''parse the class name of an image.
		The folder name of the image is treated as the class name.'''
		strlist = imagepath.split('/')
		classname = strlist[-2]
		return classname
	
	
	def imagePredict(self):
		'''predict the class name of the given image.'''
		# set the status bar.
		self.statusBar().showMessage('Predicting...')
		# calculate the bottleneck values.
		bottleneck_values = self.CalculateBottlenecks(self.imagepath)
		bottleneck_values = bottleneck_values[np.newaxis, :]
		# print(bottleneck_values.shape)
		# print(type(bottleneck_values))
		# calculate the class ID.
		classID = self.predict(bottleneck_values)
		self.PrdText.setText(str(classID))
		# set the status bar.
		self.statusBar().showMessage('Prediction OK!')
		
		# calculate the counts and prediction accuracy.
		self.totalCount = self.totalCount + 1
		self.TotalCount.setText(str(self.totalCount))
		
		if self.classID == classID:
			self.rightCount = self.rightCount + 1
			self.RightCount.setText(str(self.rightCount))
		else:
			self.wrongCount = self.wrongCount + 1
			self.WrongCount.setText(str(self.wrongCount))
		
		self.accuracy = self.rightCount/self.totalCount
		accuracy = format(self.accuracy*100, '0.2f')
		accuracy = str(accuracy) + '%'
		self.Accuracy.setText(accuracy)
		
		
if __name__ == '__main__':
    
	# Create the UI Window.
	app = QtWidgets.QApplication(sys.argv)
	myWindow = MyWindow()
	myWindow.show()
	sys.exit(app.exec_())
	