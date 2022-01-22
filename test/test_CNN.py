import unittest
import sys

from ../CNN import normalize
from ../input_dati import read_dataset, import_csv
from ../data_augmentation import VolumeAugmentation

class TestCNN(unittest.TestCase):
    """
    Class for testing the functionalities of our code
    """
    def setUp(self):
        """
        Set up the test
        """
        self.NAD=144
        self.NCTRL=189
        self.dataset_path_AD_ROI = "AD_CTRL/AD_ROIL"
        self.dataset_path_CTRL_ROI = "AD_CTRL/CTRL_ROIL"
        self.dataset_path_metadata = "AD_CTRL_metadata_labels.csv"
        _, _, self.dic = import_csv(self.dataset_path_metadata)
        self.x, self.y, self.fnames_AD, self.fnames_CTRL, self.file_id, self.age =read_dataset(self.dataset_path_AD_ROI, self.dataset_path_CTRL_ROI, self.dic)
        self.volume = VolumeAugmentation(self.x, self.y, shape=(self.x))

    def test_shape(self):
        self.assertEqual( (self.x.shape[0]), len(self.y) )
        self.assertEqual( (len(self.x.shape)), 4)
        self.assertEqual( (len(self.y.shape)), 1)

    def test_len(self):
        self.assertEqual( len(self.fnames_AD), self.NAD)
        self.assertEqual( len(self.fnames_CTRL), self.NCTRL)
        self.assertEqual( len(self.file_id), len(self.age))


    def test_metadata(self):
        df, head = import_csv(self.dataset_path_metadata)
        #count the entries grouped by the diagnostic group
        print(df.groupby('DXGROUP')['ID'].count())

    def test_max_min (self):
        self.assertEqual( (self.y.max()) , 1)
        self.assertEqual( (self.y.min()) , 0)
        nx=normalize(self.x)
        self.assertEqual( (nx.max()) , 1)
        self.assertEqual( (nx.min()) , 0)

    def test_augmentation(self):
        self.assertEqual( len(self.volume.augment()[1]), len(self.y))
        self.assertEqual( self.volume.augment()[0].shape, self.x.shape)

if __name__=='__main__':
   unittest.main(exit=not sys.flags.interactive)
