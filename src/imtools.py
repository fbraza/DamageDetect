import cv2
from typing import Tuple


def generate_blob(image, scale: float = 1/255,
                  size: Tuple[int] = (416, 416),
                  mean: int = 0,
                  crop: bool = False):
    """
    Function that generate an image blob that will feed our yolo predictons

    Returns:
    --------
    cv2.dnn.blobFromImage() -> retval
    """
    return cv2.dnn.blobFromImage(image, scale, size, mean, crop)


def crop_predictions(x: float,
                     y: float,
                     w: float,
                     h: float,
                     image):
    """
    Function that crops the image inside the predicted bounding
    box.

    Returns:
    --------
    array representing the cropped image
    """
    return image[y:y+h, x:x+w, :]
