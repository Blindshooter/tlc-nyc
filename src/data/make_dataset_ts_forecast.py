# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

def get_ts_data_fhvhv(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    # df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
    df['pickup_date'] = df['pickup_datetime'].dt.date
    # df['pickup_time'] = df['pickup_datetime'].dt.time
    # df['dropoff_date'] = df['dropoff_datetime'].dt.date
    # df['dropoff_time'] = df['dropoff_datetime'].dt.time
    # df['pickup_hour'] = df['pickup_datetime'].dt.hour
    # df['dropoff_hour'] = df['dropoff_datetime'].dt.hour
    # df['pickup_day'] = df['pickup_datetime'].dt.day
    # df['dropoff_day'] = df['dropoff_datetime'].dt.day
    # df['pickup_month'] = df['pickup_datetime'].dt.month
    # df['dropoff_month'] = df['dropoff_datetime'].dt.month
    # df['pickup_year'] = df['pickup_datetime'].dt.year
    # df['dropoff_year'] = df['dropoff_datetime'].dt.year
    # df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    # df['dropoff_weekday'] = df['dropoff_datetime'].dt.weekday
    # df['pickup_weekday_name'] = df['pickup_datetime'].dt.day_name()
    # df['dropoff_weekday_name'] = df['dropoff_datetime'].dt.day_name()
    # df['pickup_week'] = df['pickup_datetime'].dt.week
    # df['dropoff_week'] = df['dropoff_datetime'].dt.week
    # df['pickup_quarter'] = df['pickup_datetime'].dt.quarter
    # df['dropoff_quarter'] = df['dropoff_datetime'].dt.quarter
    df = df.groupby(['pickup_date', 'PULocationID'])['hvfhs_license_num'].count().reset_index()

    # rename column 'hvfhs_license_num' to 'count'
    df.rename(columns={'hvfhs_license_num': 'count'}, inplace=True)

    return df


@click.command()
@click.option('--input_filepath', default='data/raw', type=click.Path(exists=True), help='Input file path (default: data/raw)')
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
    # files = [f for f in input_filepath if f.__contains__('fhvhv_tripdata')]
    files = [f for f in input_filepath.glob('*') if 'fhvhv_tripdata' in f.name]
    res = pd.DataFrame()

    for file in files:
        logger.info(f'file: {file}')
        df_month = pd.read_parquet(file)
        # df_month = pd.read_parquet('../data/raw/'+file)
        df_month = get_ts_data_fhvhv(df_month)
        res = pd.concat([res, df_month], axis=0)
        print(df_month.shape)

    logger.info(f'final shape: {res.shape}')

    res.sort_values(by=['pickup_date', 'PULocationID'], inplace=True)
    res = res[res['PULocationID'].notna()]

    # turn irregular time series into regular time series
    start_date = res['pickup_date'].min()
    end_date = res['pickup_date'].max()

    date_range = pd.date_range(start_date, end_date)

    result = res.set_index('pickup_date').groupby('PULocationID')['count'].apply(lambda x: x.reindex(date_range,fill_value = 0)).reset_index()
    result.columns = ['PULocationID', 'pickup_date', 'count']

    result.to_parquet(output_filepath/'fhvhv_ts_forecast.parquet')

    logger.info(f'Saved file: {output_filepath}/fhvhv_ts_forecast.parquet')

    result_with_na = res.set_index('pickup_date').groupby('PULocationID')['count'].apply(lambda x: x.reindex(date_range)).reset_index()
    result_with_na.columns = ['PULocationID', 'pickup_date', 'count']
    result_with_na.to_parquet(output_filepath/'fhvhv_ts_forecast_with_na.parquet')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
