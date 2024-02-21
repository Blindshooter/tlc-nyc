# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

@click.command()
@click.option('--input_filepath', default='data/processed/fhvhv_ts_forecast.parquet', type=click.Path(exists=True), 
              help='Input file path (default: data/processed/fhvhv_ts_forecast.parquet)')
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
    df = pd.read_parquet(input_filepath)

    # create date features
    df['pickup_date'] = pd.to_datetime(df['pickup_date'])
    df['day_of_week'] = df['pickup_date'].dt.dayofweek
    df['day_of_month'] = df['pickup_date'].dt.day
    df['month'] = df['pickup_date'].dt.month
    # df['year'] = df['date'].dt.year
    # df['week_of_year'] = df['pickup_date'].dt.weekofyear
    # df['quarter'] = df['date'].dt.quarter
    # df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x > 4 else 0)

    # lag features
    df['lag_1'] = df.groupby(['PULocationID'])['count'].shift(1)
    df['lag_7'] = df.groupby(['PULocationID'])['count'].shift(7)

    # create rolling features per location
    df['rolling_mean_3'] = df.groupby(['PULocationID'])['lag_1'].rolling(3).mean().reset_index(0, drop=True)
    df['rolling_mean_7'] = df.groupby(['PULocationID'])['lag_1'].rolling(7).mean().reset_index(0, drop=True)
    df['rolling_mean_14'] = df.groupby(['PULocationID'])['lag_1'].rolling(14).mean().reset_index(0, drop=True)
    df['rolling_mean_30'] = df.groupby(['PULocationID'])['lag_1'].rolling(30).mean().reset_index(0, drop=True)

    # same for median
    df['rolling_median_3'] = df.groupby(['PULocationID'])['lag_1'].rolling(3).median().reset_index(0, drop=True)
    df['rolling_median_7'] = df.groupby(['PULocationID'])['lag_1'].rolling(7).median().reset_index(0, drop=True)
    df['rolling_median_14'] = df.groupby(['PULocationID'])['lag_1'].rolling(14).median().reset_index(0, drop=True)
    df['rolling_median_30'] = df.groupby(['PULocationID'])['lag_1'].rolling(30).median().reset_index(0, drop=True)

    # same for std
    df['rolling_std_3'] = df.groupby(['PULocationID'])['lag_1'].rolling(3).std().reset_index(0, drop=True)
    df['rolling_std_7'] = df.groupby(['PULocationID'])['lag_1'].rolling(7).std().reset_index(0, drop=True)
    df['rolling_std_14'] = df.groupby(['PULocationID'])['lag_1'].rolling(14).std().reset_index(0, drop=True)
    df['rolling_std_30'] = df.groupby(['PULocationID'])['lag_1'].rolling(30).std().reset_index(0, drop=True)

    # same for min
    df['rolling_min_3'] = df.groupby(['PULocationID'])['lag_1'].rolling(3).min().reset_index(0, drop=True)
    df['rolling_min_7'] = df.groupby(['PULocationID'])['lag_1'].rolling(7).min().reset_index(0, drop=True)
    df['rolling_min_14'] = df.groupby(['PULocationID'])['lag_1'].rolling(14).min().reset_index(0, drop=True)
    df['rolling_min_30'] = df.groupby(['PULocationID'])['lag_1'].rolling(30).min().reset_index(0, drop=True)

    # same for max
    df['rolling_max_3'] = df.groupby(['PULocationID'])['lag_1'].rolling(3).max().reset_index(0, drop=True)
    df['rolling_max_7'] = df.groupby(['PULocationID'])['lag_1'].rolling(7).max().reset_index(0, drop=True)
    df['rolling_max_14'] = df.groupby(['PULocationID'])['lag_1'].rolling(14).max().reset_index(0, drop=True)
    df['rolling_max_30'] = df.groupby(['PULocationID'])['lag_1'].rolling(30).max().reset_index(0, drop=True)

    # create expanding features per location
    df['expanding_mean'] = df.groupby(['PULocationID'])['lag_1'].expanding().mean().reset_index(0, drop=True)
    df['expanding_median'] = df.groupby(['PULocationID'])['lag_1'].expanding().median().reset_index(0, drop=True)
    df['expanding_std'] = df.groupby(['PULocationID'])['lag_1'].expanding().std().reset_index(0, drop=True)
    df['expanding_min'] = df.groupby(['PULocationID'])['lag_1'].expanding().min().reset_index(0, drop=True)
    df['expanding_max'] = df.groupby(['PULocationID'])['lag_1'].expanding().max().reset_index(0, drop=True)

    # create differencing features per location
    df['diff_1'] = df.groupby(['PULocationID'])['lag_1'].diff(1)
    df['diff_7'] = df.groupby(['PULocationID'])['lag_1'].diff(7)

    # create cyclic features from weekday
    df['day_of_week_sin'] = np.sin(df.day_of_week * (2. * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df.day_of_week * (2. * np.pi / 7))

    # create cyclic features from month
    df['day_of_week_sin'] = np.sin(df.day_of_month * (2. * np.pi / 31))
    df['day_of_week_cos'] = np.cos(df.day_of_month * (2. * np.pi / 31))

    # remove 1st month as it has missing values
    df = df[df['pickup_date'] >= df['pickup_date'].min() + pd.DateOffset(months=1)]

    # save the data
    df.to_parquet(output_filepath / 'fhvhv_ts_forecast_features.parquet')
    logger.info(f'saved data to {output_filepath / "fhvhv_ts_forecast_features.parquet"}')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
