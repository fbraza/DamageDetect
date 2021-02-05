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
        aug_data: A.Compose = pipeline(image=image, bboxes=points, class_labels=labels)
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
        Function defined
            - to save the augmented / tranformed image on the disk in a directory
            - Process the labels and transformed coordinates into an numpy array
            - Save the data encapsulated in the array in a txt file respecting the
            yolo format
        """
        # save images
        image_path = self.destin / image_name
        cv2.imwrite("{}".format(image_path), image)
        #
        points = np.array(points)
        new_yolo_bbox = np.insert(new_coordinates, 0, labels, 1)
        np.savetxt(f"{path_to_save_data}/{yolo_name}",
                new_yolo_bbox, ["%i", "%f", "%f", "%f", "%f"])


def augment_and_save(path_to_get_data, path_to_save_data,
                     number_of_tranformation=3):
    """
    Function defined to apply an image / rounding boxes transformation pipeline
    and save the corresponding files.
    Args:
    -----
    - path_to_get_data: str, the folder path where untouched data is
    - path_to_save_data: str, the folder path where to save augmented data
    - number_of_transformation: int, number of transformation to perform
    Returns:
    --------
    -  None
    """
    images_names, yolo_names = get_images_and_box_files_names(path_to_get_data)
    augmentation_pipeline = A.Compose(
            [A.Resize(416, 416),
             A.Equalize(by_channels=True),
             A.HorizontalFlip(p=0.5)],
            A.BboxParams('yolo', ['class_labels'])
            )
    # Iterate through each image
    for idx, name in enumerate(images_names):
        image_path = path_to_get_data + '/' + name
        image = cv2.imread(image_path)
        yolo_file_path = path_to_get_data + '/' + yolo_names[idx]
        labels, coordinates = get_labels_and_coordinates(yolo_file_path)
        # Generate x tranformation of the images
        for i in tqdm(range(number_of_tranformation)):
            new_image_name, new_yolos_name = set_new_files_names(
                                                name, i, "jpg", "txt"
                                                )
            # Catch error due to unproper labelling
            try:
                new_image, new_coordinates, labels = get_data_from_pipeline(
                                                        augmentation_pipeline,
                                                        image, coordinates,
                                                        labels
                                                        )
            except ValueError as e:
                print("**** Error Message ****\n")
                print(f"{e}\n")
                print(f"""Invalid transformation of box:
                          {str(new_coordinates)}\n""")
                print(f"Image: {new_image_name}\n")
                continue
            # Save each image to jpg with its corresponding coordinates
            save_image_bbox_data(path_to_save_data, new_image, new_image_name,
                                 new_coordinates, labels, new_yolos_name)
