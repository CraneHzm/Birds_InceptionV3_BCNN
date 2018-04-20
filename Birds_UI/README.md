# Copyright (c) 2018/4/20 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.


# The UI for classifying Birds.


# Directories & Files:
'TestImages' directory: stores a part of the original test images for UI test.
'Models' directory: stores our trained BCNN model.
'InceptionV3.py': Calculate the bottleneck values of the given images using Inception V3 network.
'BCNN.py': Calculate the class ID of the input bottleneck values.
'UI.py': Set up the UI.
'UI.ui': 'ui' format of 'UI.py'.
'Main.py': the main function, run it to launch the UI.

# Environments:
Python 3.6+
pyqt5
tensorflow 1.7+

# Usage:

Step 1: Run 'Main.py' to launch the UI.
Step 2: Click 'Select Image' button to select the image to be classified.
Step 3: Click 'Predict' button to make a prediction.
Step 4: Click 'Exit' button to exit the UI.