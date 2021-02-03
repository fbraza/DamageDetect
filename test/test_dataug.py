import unittest
from ..src.dataug import Augmentor


class TestIoMethods(unittest.TestCase):

    def test_get_images_and_box_files_names(self):
        aug = Augmentor()