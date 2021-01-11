import cv2


def generate_blob(image, scale=1/255, size=(416, 416), mean=0, crop=False):
    """
    Function that generate an image blob that will feed our yolo predictons

    Args:
    -----
    image: array representing the image
    scale: scaling factor of the resutling blob
    size: size of the output image
    mean: value to be substracted from rgb channels
    crop: whether or not the image will be cropped

    Returns:
    --------
    cv2.dnn.blobFromImage() -> retval
    """
    return cv2.dnn.blobFromImage(image, scale, size, mean, crop)


def crop_predictions(x, y, w, h, image):
    """
    Function that crops the image inside the predicted bounding
    box.

    Args:
    -----
    x: center position on the x axis
    y: center position on the y axis
    w: width of the bouding box
    h: height of the bounding box
    image: array representing the image

    Returns:
    --------
    array representing the cropped image
    """
    return image[y:y+h, x:x+w, :]
