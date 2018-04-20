# Copyright (c) 2018/4/19 Hu Zhiming JimmyHu@pku.edu.cn All Rights Reserved.
# The tensorflow CNN & Bilinear CNN codes for Birds CUB-200-2011 Dataset.

# Directories & Files:

'DataPreprocess' directory: stores the codes to preprocess the original images.
'Dataset' directory: stores the TFRecord files created in the DataPreprocess directory.
'Models' directory: stores our trained models.
'Results' directory: stores the learning curves of our model.


# Environments:
Python 3.6+

tensorflow 1.7+

# Usage:

Step 1: Check the 'Dataset/' directory to confirm whether the TFRecord files exist. 
If not, run the codes in 'DataPreprocess/' to create the TFRecord files.

Step 2: Run 'Birds.py' & 'Birds_BCNN.py' to test the model.
The accuracy of 'Birds.py' on the test set is 48.02%. 
The accuracy of 'Birds_BCNN.py' on the test set is 53.30%. 
If the models do not exit, you can uncomment the training code in the main function to retrain the model.
