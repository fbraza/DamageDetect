import unittest
import os
from pathlib import Path
from src.dataug import Augmentor
from typing import List


class TestAugmentorMethods(unittest.TestCase):

    def setUp(self):
        self.aug: Augmentor = Augmentor(
            Path("test/test_source/"),
            Path("test/test_destin/")
        )

    def test_get_images_and_box_files_names(self):
        result: List[str] = self.aug.get_file_names()
        value: List[str] = sorted(['7', '8', '9', '10', '11'])
        self.assertEqual(value, result)

    def test_get_labels_and_coordinates(self):
        file: Path = self.aug.source / "7.txt"
        labels: List[int] = [0, 1, 1, 1]
        points: List[List[float]] = [
            [0.506172, 0.450620, 0.760017, 0.443467],
            [0.383168, 0.329990, 0.041267, 0.026934],
            [0.524757, 0.644043, 0.284931, 0.027734],
            [0.536372, 0.262646, 0.598333, 0.049785]
            ]
        self.aug.get_labels_and_coordinates(file)
        self.assertEqual((self.aug.labels, self.aug.bboxes), (labels, points))

    def test_augment_and_save(self):
        self.aug.augment_and_save(
            self.aug.source,
            self.aug.destin,
            number_of_tranformation=2
        )
        file_count = len(os.listdir(self.aug.destin))
        self.assertEqual(file_count, 20)

    def tearDown(self) -> None:
        return super().tearDown()


if __name__ == '__main__':
    unittest.main()
