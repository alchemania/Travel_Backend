# python packages
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import django
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from pypots.imputation import Transformer
from pypots.optim import Adam
from sklearn.preprocessing import MinMaxScaler

from utils.dataprocess import interpolation

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TravelServer.settings')
django.setup()

from django_pandas.io import read_frame

logger = logging.getLogger(__name__)

# django imports
from api.models import *
from django.db.models import Max
from TravelServer import settings

# self-defined module import
from utils.spider import HKVisitorsSpider, ShHotelSpider, ShVisitorsSpider
from utils.dataprocess import melt, cut


def crawl_sh_hotel(*args, **kwargs):
    logger.info("Starting the hotel spider task")
    try:
        tasks = DbSpider.objects.filter(unique_id__contains=ShHotelSpider.identifier).all()
        tasks = read_frame(tasks).to_dict(orient='list')
        tasks = dict(zip(tasks['unique_id'], tasks['url']))
        spider = ShHotelSpider('update', tasks=tasks)
        spider.run()
        logger.info("Hotel spider task fetching completed successfully")

        if spider.tasks:
            logger.info("Hotel spider new data fetched")
            objs = [DbSpider(unique_id=uid, url=url) for uid, url in spider.tasks]
            DbSpider.objects.bulk_create(objs, ignore_conflicts=True)
            logger.info("New spider tasks added to database")

            data = spider.data()
            instances = [
                DbShHotel(
                    date=row[ShHotelSpider.pd_columns[0]],
                    avg_rent_rate=row[ShHotelSpider.pd_columns[1]],
                    avg_rent_rate_5=row[ShHotelSpider.pd_columns[2]],
                    avg_price=row[ShHotelSpider.pd_columns[3]],
                    avg_price_5=row[ShHotelSpider.pd_columns[4]],
                )
                for index, row in data.iterrows()
            ]
            DbShHotel.objects.bulk_create(instances, ignore_conflicts=True)
            logger.info("Hotel data inserted into database")
            return 0
        else:
            logger.info("Hotel spider task no new data fetched")
            return 0
    except Exception as e:
        logger.error(f"An error occurred in the hotel spider task: {str(e)}")
        return -1


def crawl_sh_visitors(*args, **kwargs):
    logger.info("Starting the hotel spider task")
    try:
        tasks = DbSpider.objects.filter(unique_id__contains=ShVisitorsSpider.identifier).all()
        tasks = read_frame(tasks).to_dict(orient='list')
        tasks = dict(zip(tasks['unique_id'], tasks['url']))
        spider = ShHotelSpider('update', tasks=tasks)
        spider.run()
        logger.info("sh visitors spider task fetching completed successfully")

        if spider.tasks:
            logger.info("sh visitors spider new data fetched")
            objs = [DbSpider(unique_id=uid, url=url) for uid, url in spider.tasks.items()]
            DbSpider.objects.bulk_create(objs, ignore_conflicts=True)
            logger.info("New spider tasks added to database")

            data = spider.data()
            instances = [
                DbShvisitorsMonthly(
                    DATE=row[ShVisitorsSpider.pd_columns[0]],
                    FOREIGN=row[ShVisitorsSpider.pd_columns[1]],
                    HM=row[ShVisitorsSpider.pd_columns[2]],
                    TW=row[ShVisitorsSpider.pd_columns[3]],
                )
                for index, row in data.iterrows()
            ]
            DbShvisitorsMonthly.objects.bulk_create(instances, ignore_conflicts=True)
            logger.info("SH Visitor data inserted into database")
            return 0
        else:
            logger.info("SH Visitor spider task no new data fetched")
            return 0
    except Exception as e:
        logger.error(f"An error occurred in the hotel spider task: {str(e)}")
        return -1


def crawl_hk_visitors(*args, **kwargs):
    logger.info("Starting the HK visitors spider task")
    try:
        spider = HKVisitorsSpider()
        spider.run()
        logger.info("HK visitors spider task fetching completed successfully")
        max_date_db = DbHkVisitors.objects.aggregate(max_date=Max('date'))['max_date'].date()
        max_date_spd = spider.data['date'].max()
        if max_date_spd > max_date_db or max_date_db is None:
            logger.info("HK visitors spider new data fetched")
            data = spider.data[spider.data['date'] > max_date_db]
            instances = [
                DbHkVisitors(
                    date=row['date'],
                    HK_airport_entry=row['HK_airport_entry'],
                    CN_airport_entry=row['CN_airport_entry'],
                    global_airport_entry=row['global_airport_entry'],
                    airport_entry=row['airport_entry'],
                    HK_airport_departure=row['HK_airport_departure'],
                    CN_airport_departure=row['CN_airport_departure'],
                    global_airport_departure=row['global_airport_departure'],
                    airport_departure=row['airport_departure']
                )
                for index, row in data.iterrows()
            ]
            DbHkVisitors.objects.bulk_create(instances, ignore_conflicts=True)
            logger.info("HK visitors data updated in database")
            return True
        else:
            logger.info("HK visitors spider no new data fetched")
            return False
    except Exception as e:
        logger.error(f"An error occurred in the HK visitors spider task: {str(e)}")


# test passed
def impute_hk_visitors(*args, **kwargs):
    # fetch Data
    scaler = MinMaxScaler()
    df_ori = read_frame(DbHkVisitors.objects.all(), index_col='date', datetime_index=True)

    # MinMax for each row
    df = pd.DataFrame(
        data=scaler.fit_transform(df_ori.values),
        columns=df_ori.columns,
        index=df_ori.index
    )

    # Reindex
    index = pd.date_range(start='2019-01-01', end=df_ori.index.max(), freq='D')
    df_ori = df_ori.reindex(index)
    df = df.reindex(index)

    # sample&features
    num_samples = df.values.shape[0]
    num_features = df.values.shape[1]  # 减去date

    transformer = Transformer(
        n_steps=num_samples,
        n_features=num_features,
        n_layers=4,
        d_model=192,
        d_ffn=128,
        n_heads=6,
        d_k=32,
        d_v=64,
        dropout=0.1,
        attn_dropout=0,
        epochs=300,
        optimizer=Adam(lr=1e-3),
        device=torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        patience=50,
    )
    ds = {"X": df.values.reshape(1, num_samples, -1)}
    transformer.fit(ds)

    # impute and convert back
    impute = pd.DataFrame(
        scaler.inverse_transform(transformer.impute(ds).squeeze()),
        columns=df.columns,
        index=df.index)
    # impute = impute.apply(lambda col: scalers[col.name].inverse_transform(col.values.reshape(-1, 1)).flatten())
    impute = impute.astype('int')
    impute = pd.DataFrame({
        col: np.where(
            (np.isnan(df.values) ^ np.isnan(impute.values)).reshape(num_samples, num_features)[:, 0],
            impute[col],
            df_ori[col]
        ) for col
        in
        df_ori.columns
    }, index=df.index)

    # 操作数据库
    instances = [
        DbHkVisitorsImputed(
            date=row['date'],
            HK_airport_entry=row['HK_airport_entry'],
            CN_airport_entry=row['CN_airport_entry'],
            global_airport_entry=row['global_airport_entry'],
            airport_entry=row['airport_entry'],
            HK_airport_departure=row['HK_airport_departure'],
            CN_airport_departure=row['CN_airport_departure'],
            global_airport_departure=row['global_airport_departure'],
            airport_departure=row['airport_departure']
        )
        for index, row in impute.iterrows()
    ]
    DbHkVisitorsImputed.objects.bulk_create(instances, update_conflicts=True)
    logger.info("HK visitors data updated in database")


def interpolate_sh_visitors(*args, **kwargs):
    sh = DbShvisitorsMonthly.objects.all()
    hk = DbHkVisitorsImputed.objects.all()

    interpolate = interpolation(sh_ori=sh, hk_ori=hk, noise=0.15)

    # 操作数据库
    instances = [
        DbShvisitorsDaily(
            DATE=row['date'],
            FOREIGN=row['global_entry'],
            HM=row['hkmo_entry'],
            TW=row['tw_entry'],
        ) for index, row in interpolate.iterrows()
    ]
    DbShvisitorsMonthly.objects.bulk_create(instances, update_conflicts=True)
    logger.info("Sh visitors Interpolated data updated in database")


def _get_latest_model(directory, *args, **kwargs):
    logger.info("Searching for the latest model directory")
    max_num = -1
    max_folder = None
    pattern = re.compile(r'modelgroup_(\d+)')
    try:
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                match = pattern.match(item)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        max_folder = full_path
        logger.info(f"Latest model directory found: {max_folder}")
        return max_folder
    except Exception as e:
        logger.error(f"An error occurred while searching for the latest model directory: {str(e)}")
        return None


def train(*args, **kwargs):
    logger.info("Starting auto-training task")
    try:
        base_dir = os.path.join(settings.BASE_DIR, 'models')
        model_dir = _get_latest_model(base_dir)
        df = DbShvisitorsDaily.objects.all()
        df = read_frame(df, index_col='DATE')
        df = cut(df)
        df = melt(df)
        nf = NeuralForecast.load(model_dir, verbose=False)
        nf.fit(df=df)
        nf.save(str(Path(model_dir).parent.joinpath(f'modelgroup_{datetime.now().strftime("%Y%m%d%H%M%S")}')))
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in the auto-training task: {str(e)}")


def predict(*args, **kwargs):
    logger.info("Starting auto-prediction task")
    try:
        base_dir = os.path.join(settings.BASE_DIR, 'models')
        model_dir = _get_latest_model(base_dir)
        db_data = DbShvisitorsDaily.objects.all()
        dataset = melt(cut(read_frame(db_data, index_col='DATE')))
        iters = 6  # 3*60=180
        nf = NeuralForecast.load(model_dir, verbose=False)
        model_names = ['Auto' + cls.__class__.__name__ for cls in nf.models]
        df_pred = dataset.copy()
        for i in range(iters):
            step = nf.predict(df_pred).reset_index()
            step['y'] = step['AutoPatchTST']
            step = step.drop(columns=model_names)
            df_pred = pd.concat([df_pred, step], axis=0)
        instances = [
            DbShvisitorsDailyPredicted(
                date=row['date'],
                FOREIGN=row['global_entry'],
                HM=row['hkmo_entry'],
                TW=row['tw_entry']
            )
            for index, row in df_pred.iterrows()
        ]
        DbHkVisitorsImputed.objects.bulk_create(instances, update_conflicts=True)
        logger.info("Prediction data updated in database")
    except Exception as e:
        logger.error(f"An error occurred in the auto-prediction task: {str(e)}")


if __name__ == '__main__':
    # auto_hotel_spider()
    # auto_hkvisitors_spider()
    # autotrain()
    # autopredict()

    impute_hk_visitors()
