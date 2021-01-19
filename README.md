# State predictions of EDP electronic cabinets

Repository containing the full workflow to:

- process, split and prepare data for object detection training in YOLOv4 (`src/dataug.py`, `src/datasplit.py`). 
- Harness the training weights for prediction and cropping of the desired object (`src/predict.py`)
- Classify the state of EDP electronic cabinets based on cropped images

## Installation

- **Requirements**
  - Ensure that you have `git` & `dvc` installed in your machine ([git installation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), [dvc installation](https://dvc.org/doc/install)).
  - Ensure that python >= 3.6 is installed
- **Repository**

To use the repository go to your working folder, open a terminal and run:

```bash
git clone https://gitlab.com/fbraza/edp-altran.git
```

You should be prompted to enter your `gitlab` login and password. Next to install dependencies run:

```bash
pip install -r requirements.txt
```

Finally once you have set up everything, go to the root folder of the repository, open a terminal and run:

```bash
dvc pull
```

This will get the last version of the data for this project.

- **YOLO training**

For object detection training we used the version 4 of [YOLO](https://github.com/AlexeyAB/darknet). Don't be tricked by the naming `Darknet` which is actually the name of the neural network architecture.
If you want to re-run the training from scratch with YOLO, please first check if you have the `C` environment installed. Also if you want to have the possibility to use the
GPU check that `cuda` and `nvidia` drivers are installed and up-to-date. Then to use YOLO on your computer, you need to clone the repository:

```bash
git clone https://github.com/AlexeyAB/darknet
```

Next you need to edit the value of some `C` constants in the `Makefile`.

```
GPU=0        # change to 1 to speed on graphic card
CUDNN=0      # change to 1 to speed on graphic card
CUDNN_HALF=0 # change to 1 to further speed up on graphic cards (only works for powerful GPU)
OPENCV=0     # change to 1 if you use openCV
```

To modify these values y can use a command similar to the one below:

```bash
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

Once done you can compile the code using:

```
make
```

Before running the training you need to configure some parameters. In `darknet/cfg/` you have a list of configuration file with the extension `.cfg`. These files defines the way the
architecture will be used depndeing on whether you use full or lite YOLO models. In our case we used the yolov4.cfg and edit it as followed:
- `batch = 64` and `subdivisions = 16` for ultimate results. If you run into memory issues then up `subdivisions` to `32`.
- `width = 416`, `height = 416` (these should be multiple of 32, 416 is standard)
- Make the rest of the changes to the `.cfg` file based on how many classes you have to train your detector with.
  - to determine the number of `max_batches` use the following formula: `max_batches = (# of classes) * 2000` (do not go below 6000 anyway). Modify the value accordingly in the three `yolo` layers
  - to determine the number of `filters` use the following formula: `filters = (# of classes + 5) * 3`. Modify the value accoridngly in the three `convolutional` layers

## Usage

The interface of the repository has been encapsulated into a CLI application.  Running the following command:

```bash
python app.py --help
```

will output the helper documentation.

```bash
Usage: app.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  predict-and-output  localize boxes and scrap them out
  prepare             load img, split into train-val and prepare for yolo
  transform           Transform and augment a set of raw images
```

To get help on any commands and read the required arguments run

```bash
python app.py [COMMAND] --help
```

Example:

```bash
# python app.py prepare --help
Usage: app.py prepare [OPTIONS]

  load img, split into train-val and prepare for yolo

Options:
  --path_in TEXT        Path of transformed images  [required]
  --split-factor FLOAT  Factor to split into training and validation sets
  --help                Show this message and exit.
```

To launch a new training with yolov4, prepare your data accoridngly (cloning the repository, getting the data with dvc and run the transform and prepare commands from our CLI app).
Then move the content of the `data/yol_images` to `darknet\data` and run:

```bash
./darknet detector train data/obj.data cfg/yolo4-custom.cfg yolov4.conv.137 -map
```

The training should take from several hours to days depending on the computer power you have access to and the number of classes you train your model with. Once finished run the following 
command to get the metrics on a validation set.


## ToDo

- [ ] Add the classification `src/classify.py` to the CLI app
- [ ] Encapsulate the full process into a pipeline

# Author

Faouzi Braza, Expertise Center of AI & Analytics, Altran (Portugal)

[Contact email](faouzi.braza@altran.com)

[Repository link](https://gitlab.com/fbraza/edp-altran)
