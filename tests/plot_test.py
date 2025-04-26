import os
import tempfile
import unittest

import pandas as pd

from plots import load_and_prepare


class TestSubregionAnalysis(unittest.TestCase):
    def setUp(self):
        # slice-based
        self.tmp1 = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.tmp1.write("""mouse_id,group,subregion,slice1,slice2,slice3
m1,naive control,r1,1,2,3
m2,rotarod control,r1,4,5,6
m3,rotarod,r1,7,8,9
""")
        self.tmp1.flush()
        self.tmp1.close()
        # roi_rate-based
        self.tmp2 = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.tmp2.write("""mouse_name,area_name,sub_area_name,roi_rate
m1,area1,center,0.5
m1,area1,left,0.3
""")
        self.tmp2.flush()
        self.tmp2.close()

    def tearDown(self):
        os.unlink(self.tmp1.name)
        os.unlink(self.tmp2.name)

    def test_load_prepare_slice(self):
        df = load_and_prepare(self.tmp1.name)
        self.assertEqual(len(df), 3)

    def test_load_prepare_roi(self):
        df = load_and_prepare(self.tmp2.name)
        self.assertTrue('mean_fluorescence' in df.columns)
        self.assertTrue('slice_number' in df.columns)

    def test_multiple_csv(self):
        df1 = load_and_prepare(self.tmp1.name)
        df2 = load_and_prepare(self.tmp2.name)
        df_all = pd.concat([df1, df2], ignore_index=True)
        self.assertEqual(len(df_all), 5)