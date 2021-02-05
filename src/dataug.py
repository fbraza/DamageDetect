import os
import re
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, TextIO, Tuple, Pattern


class Augmentor:
    def __init__(
        self,
        source: Path,
        destin: Path
    ) -> None:
        self.source = source
        self.destin = destin
        self.labels = None
        self.bboxes = None

    def get_images_and_box_files_names(
        self
    ) -> List[str]:
        """
        Returns root names of the img and bboxes files.
        """
        pattern: Pattern[str] = r"[^\\.txt|^\\.jpeg]+"
        files: List(str) = os.listdir(self.source)
        names: Dict = {re.match(pattern, name)[0]: True for name in files}
        names: List = list(names.keys())
        return sorted(names)

    def get_labels_and_coordinates(
        self,
        bbox_file: Path
    ) -> Tuple:
        """
        Labels and points are processed from file, splitted
        and returned in two lists
        """
        file: TextIO = open(bbox_file, "r")
        labels: List[int] = []
        points: List = []
        for line in file.readlines():
            split_line: List[str] = line.split()
            label: int = int(split_line[0])
            coord: List[float] = list(map(float, split_line[1:]))
            labels.append(label)
            points.append(coord)
        file.close()
        self.labels = labels
        self.points = points

    def get_data_from_pipeline(
            self,
            pipeline: A.Compose,
            image: np.ndarray,
            points: List[float],
            labels: List[int]
    ) -> Tuple(np.ndarray, List[List[float]], List[int]):
        """
        Apply the data augmentation pipeline and retrieve image,
        bounding boxes coordinates and corresponding class labels.
        Returns a Tuple['ndarray', List['ndarray'], List[float]]
        """
        aug_data: A.Compose = pipeline(
            image=image,
            bboxes=points,
            class_labels=labels
        )
        return aug_data['image'], aug_data['bboxes'], aug_data['class_labels']

    def save_image_bbox_data(
        self,
        image: np.ndarray,
        image_name: str,
        points: List[float],
        labels: List[int],
        point_name: str
    ) -> None:
        """
        Save the image and its corresponding bboxes data
        """
        # save images
        image_path: Path = self.destin / image_name
        bboxe_path: Path = self.destin / point_name
        cv2.imwrite("{}".format(image_path), image)
        # recreate bounding boxes and save in text file the data
        points: np.ndarray = np.array(points)
        bboxes: np.ndarray = np.insert(points, 0, labels, 1)
        np.savetxt(
            {}.format(bboxe_path),
            bboxes,
            ["%i", "%f", "%f", "%f", "%f"]
        )

    def augment_and_save(
        self,
        source: Path,
        destin: Path,
        number_of_tranformation: int = 10
    ) -> None:
        """
        Apply predifined data augmentation pipeline to images
        and bounding boxes. Write and save the new files.
        """
        names = self.get_images_and_box_files_names()
        dag: A.Compose = A.Compose(
            [
                A.Resize(416, 416),
                A.Equalize(by_channels=True),
                A.HorizontalFlip(p=0.5)
            ],
            A.BboxParams('yolo', ['class_labels'])
        )
        for idx, name in enumerate(names):
            img_path: Path = source / name / ".jpg"
            txt_path: Path = source / name / ".txt"
            img: np.ndarray = cv2.imread(img_path)
            self.get_labels_and_coordinates(txt_path)
            for i in tqdm(range(number_of_tranformation)):
                new_img_name: Path = source / name / i / ".jpg"
                new_txt_name: Path = source / name / i / ".txt"
                try:
                    timg, pts, lab = self.get_data_from_pipeline(
                        dag,
                        img,
                        self.bboxes,
                        self.labels
                    )
                except ValueError as e:
                    print("**** Error Message ****\n")
                    print("{}\n".format(e))
                    print("Image: {}\n".format(new_img_name))
                    continue
                self.save_image_bbox_data(
                    self.destin,
                    timg,
                    new_img_name,
                    pts,
                    lab,
                    new_txt_name
                )
