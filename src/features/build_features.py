# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd



@click.command()
@click.option('--input_filepath', default='data/processed', type=click.Path(exists=True), help='Input file path (default: data/processed)')
@click.option('--output_filepath', default='data/processed', type=click.Path(), help='Output file path (default: data/processed)')
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)
    logger.info(f'input_filepath: {input_filepath}')
    logger.info(f'output_filepath: {output_filepath}')

    # Load the data
    df = pd.read_csv('../data/processed/', parse_dates=True)
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
