import click
import os
import cv2
from tqdm import tqdm
from src import dataug, datsplit, predict, imtools


@click.group()
def cli():
    pass


@cli.command()
@click.option("--path_in",
              required=True,
              help="Path of raw images")
@click.option("--path_out",
              required=True,
              help="Path where transformed images will be saved")
@click.option("--nbr_trans",
              default=10,
              type=click.INT,
              help="Rounds of transformation")
def transform(path_in, path_out, nbr_trans):
    """Transform and augment a set of raw images"""
    dataug.augment_and_save(path_in, path_out, nbr_trans)


@cli.command()
@click.option("--path_in",
              required=True,
              help="Path of transformed images")
@click.option("--split-factor",
              default=0.75,
              type=click.FLOAT,
              help="Factor to split into training and validation sets")
def prepare(path_in, split_factor):
    """load img, split into train-val and prepare for yolo"""
    datsplit.generate_yolo_inputs(path_in, split_factor)


@cli.command()
@click.option("--config_path",
              required=True,
              help="Path for Yolo config file")
@click.option("--weigth_path",
              required=True,
              help="Path for Yolo training weights")
@click.option("--class_path",
              required=True,
              help="Path obj.names file")
@click.option("--img_input",
              required=True,
              help="Images path for predictions")
@click.option("--output_pred",
              required=True,
              help="Predictions output path")
@click.option("--output_crop",
              required=True,
              help="Scrapped boxes output path")
@click.option("--pred_threshold",
              default=0.5,
              type=click.FLOAT,
              help="Prediction threshold for bounding boxes")
def predict_and_output(config_path,
                       weigth_path,
                       class_path,
                       img_input,
                       output_pred,
                       output_crop,
                       pred_threshold):
    """localize boxes and scrap them out"""
    # iterate through images input predict boxes and scrap them
    print("## --- Generating prediction and saving outpus --- ##")
    for image in tqdm(os.listdir(img_input)):
        predictor = predict.YoloPredictionModel(
            config_path,
            weigth_path,
            class_path
        ).set_backend_and_device()
        frame = cv2.imread("{}{}".format(img_input, image))
        blob_input = imtools.generate_blob(frame)
        predictor.ingest_input(blob_input)
        layers = predictor.get_output_layers_names()
        output = predictor._forward()
        predictor.predict_and_identify(frame, output, threshold=pred_threshold)
        cv2.imwrite("{}{}".format(output_pred, image), frame)
        try:
            cropped_img = imtools.crop_predictions(
                predictor.x_coord,
                predictor.y_coord,
                predictor.w_coord,
                predictor.h_coord,
                image=cv2.imread("{}{}".format(img_input, image))
            )
            cv2.imwrite("{}{}".format(output_crop, image), cropped_img)
        except TypeError:
            continue


if __name__ == '__main__':
    cli()
