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

## ToDo

- [ ] Add the classification `src/classify.py` to the CLI app

# Author

Faouzi Braza, Expertise Center of AI & Analytics, Altran (Portugal)

[Contact email](faouzi.braza@altran.com)