import click
from src import data_aug, data_split


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
    data_aug.augment_and_save(path_in, path_out, nbr_trans)


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
    data_split.generate_yolo_inputs(path_in, split_factor)


if __name__ == '__main__':
    cli()
