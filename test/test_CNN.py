import unittest
import sys
from CNN import read_dataset
from data_augmentation import VolumeAugmentation

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
   unittest.main(exit=not sys.flags.interactive)
