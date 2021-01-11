import cv2
import numpy as np


class YoloPredictionModel:
    def __init__(self, path_config, path_weigths, path_classes):
        """
        Instantiate an object which encapsulates all data and API necessary to
        run the predicitions

        Args:
        -----
        - configs: str, path to the .cfg file containing model configuration
        - weights: str, path to the .weights generated after training
        - classes: list, list of string containing class names

        Attr:
        -----
        - classes: list[str], list containing names of the classes
        - network: dnn module that provides yolo API with OpenCV backend
        - output_layers: list[str], list containing the names of the 3 output
          layers
        - x, y, h, w = coordinates of the predicted boxes
        """
        self.classes = self.class_names(path_classes)
        self.network = cv2.dnn.readNetFromDarknet(path_config, path_weigths)
        self.output_layers = self.get_output_layers_names()
        self.x_coord = None
        self.y_coord = None
        self.h_coord = None
        self.w_coord = None

    def class_names(self, path):
        """
        Method defined to generate a list of class names' strings

        Args:
        -----
        - path: str, path to the obj.names file

        Return:
        -------
        - None
        """
        with open(path, "rt") as file:
            return file.read().rstrip("\n").split("\n")

    def set_backend_and_device(self, device="CPU"):
        """
        Setter method to define openCV backend and the device
        used for the predicitons and calculation (CPU or GPU).
        In our case we limit the backend choice to OpenCV and the
        device choices to the use of CPU or CUDA. We leverage the
        dnn module from the cv2 library for that.

        Args:
        -----
        - device: str, choose between CPU or GPU. CPU by default

        Return:
        -------
        - self
        """
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        if device == "CPU":
            self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif device == "GPU":
            self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            raise ValueError("Wrong device choice: choose CPU or GPU")
        return self

    def ingest_input(self, blob):
        """
        Setter method defined to send input to our yolo model.

        Args:
        -----
        - blob: 4D numpy array object (images, channels, width, height)

        Return:
        -------
        - None
        """
        self.network.setInput(blob)

    def _get_layers_names(self):
        """
        Getter method to that return as a list the name of all layers
        present in our yolo neural network.

        Return:
        -------
        - list: list of string of the layers' names
        """
        return self.network.getLayerNames()

    def _get_output_layers_number(self):
        """
        Getter method to get output layers position number. The yolo model has
        three output layers.

        Return:
        -------
        - list[list[int]]: Layer position. Be careful the fisrt layer is at
        position 1. Do not forget to 0 index the values to get layers names
        """
        return self.network.getUnconnectedOutLayers()

    def get_output_layers_names(self):
        """
        Getter method to extract the output layers based on their position
        inside the neural network architecture

        Return:
        -------
        - list: return the names of the three output layers of yolo
        """
        names = self._get_layers_names()
        numbers = self._get_output_layers_number()
        return [names[i[0] - 1] for i in numbers]

    def _forward(self):
        """
        Method to return the output object of the output layers. The output
        object is a list of numpy arrays. Each numpy array is the output object
        of one output layer:
        - The first 4 columns of each array correspond to the values of the
        bounding box coordinates as floats between 0 and 1
        - The fifth column of each array correspond the box confidence
        - The rest of the columns of each array correspond to each class with
          associated probability

        Return:
        -------
        - list[floats]

        Examples:
        ---------
        In our model we are suppose to get the following:
        print((yolo.output_object()[0]).shape) => (8812, 13)
        print((yolo.output_object()[1]).shape) => (2028, 13)
        print((yolo.output_object()[2]).shape) => (507,  13)
        shape[0] => decreasing values corresponds to the bounding boxes
                    produced by the corresponding layer.
        shape[1] => [:4] bounding box coordinates, [4] box confidence, [5:]
                    class probability
        """
        return self.network.forward(self.output_layers)

    def predict_and_identify(self, image, yolo_output_objects, threshold=0.5):
        """
        Method defined to get and output predictions on video frames.
        A first pair of nested loop will get the index of the highest
        class probability and its related probability. If probability
        is superior to the threshold we save the index.
        A second loop iterate through an enumerate iterable and get the
        corresponding class in order to output it in the video frame.

        Args:
        -----
        - image: numpy array representation of the image
        - yolo_output_objects: output object of the output layers. See
        _forward() method for a more detailed description
        - threshold: float, 0.5 by default
        """
        img_heigth, img_width = image.shape[:2]
        class_index, class_proba, boxes = [], [], []
        for output_object in yolo_output_objects:
            for predictions in output_object:
                probabilities = predictions[5:]
                _index = np.argmax(probabilities)
                _proba = float(probabilities[_index])
                if _proba >= threshold:
                    # Normalize coordinates values to image size
                    w = int(predictions[2] * img_width)
                    h = int(predictions[3] * img_heigth)
                    x = int(predictions[0] * img_width)
                    y = int(predictions[1] * img_heigth)
                    point_0 = int(x - w // 2), int(y - h // 2)
                    boxes.append([*point_0, int(w), int(h)])
                    class_index.append(_index)
                    class_proba.append(_proba)
        indices = cv2.dnn.NMSBoxes(boxes, class_proba, threshold, threshold -
                                   0.3)
        # Use coordinates values to draw box and output message
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # save the coordinates for cropping as object attributes
                self.x_coord = x
                self.y_coord = y
                self.h_coord = h
                self.w_coord = w
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
                message = "edp-armario: {:.4f}".format(class_proba[i])
                cv2.putText(image, message, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
