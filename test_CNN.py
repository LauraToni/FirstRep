import unittest
import numpy as np
from cnn_da_testare import read_dataset
import os
import PIL
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')

#from data_augmentation import VolumeAugmentation

dataset_path_AD_ROI = "AD_CTRL/AD_ROIL"
dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROIL"
dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

class TestCNN(unittest.TestCase):
    """
    Class for testing the functionalities of our code
    """
    def setUp(self):
        """
        Set up the test
        """
        self.x, self.y, _, _ =read_dataset(dataset_path_AD_ROIL, dataset_path_CTRL_ROIL)

    def test_len(self):
        self.assertEqual( len(self.x[0]) , len(self.y) )


if __name__=='__main__':
   unittest.main()
