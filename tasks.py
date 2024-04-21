import eventlet

eventlet.monkey_patch()

# python packages
import os
import re

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TravelServer.settings')
django.setup()

import numpy as np
from django_pandas.io import read_frame

from celery import Celery, chain, group
from celery.schedules import crontab
from celery.utils.log import get_task_logger

# Set logging
logger = get_task_logger(__name__)

# django imports
from api.models import *
from django.db.models import Max
from TravelServer import settings

# self-defined module import
from pipeline.spider import HKVisitorsSpider, SHHotelSpider
from pipeline.model import train, predict
from pipeline.dataprocess import melt, pivot, cut

broker = 'redis://127.0.0.1:6379'
backend = 'redis://127.0.0.1:6379/0'

app = Celery('autorunner', broker=broker, backend=backend)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Accept content types
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(task_name)s/%(task_id)s] %(message)s",
    worker_log_color=False,
    task_soft_time_limit=600,  # Soft time limit in seconds for tasks
    task_time_limit=1200,  # Hard time limit in seconds for tasks
    worker_redirect_stdouts_level='INFO',
    broker_connection_retry_on_startup=True
)

# Periodic task scheduling
app.conf.beat_schedule = {
    'run_pipeline_every_hour': {
        'task': 'tasks.pipeline',
        'schedule': crontab(minute='0', hour='*'),  # Run every hour at minute 0
    }
}


@app.task
def auto_hotel_spider():
    logger.info("Starting the hotel spider task")
    try:
        tasks = DbSpider.objects.filter(unique_id__contains=SHHotelSpider.identifier).all()
        urls = [[task.unique_id, task.url] for task in tasks]
        spider = SHHotelSpider('update', tasks=urls)
        spider.run()
        logger.info("Hotel spider task fetching completed successfully")

        if spider.tasks:
            objs = [DbSpider(unique_id=uid, url=url) for uid, url in spider.tasks]
            DbSpider.objects.bulk_create(objs, ignore_conflicts=True)
            logger.info("New spider tasks added to database")

            data = spider.data()
            instances = [
                DbshHotel(
                    date=row[SHHotelSpider.pd_columns[0]],
                    avg_rent_rate=row[SHHotelSpider.pd_columns[1]],
                    avg_rent_rate_5=row[SHHotelSpider.pd_columns[2]],
                    avg_price=row[SHHotelSpider.pd_columns[3]],
                    avg_price_5=row[SHHotelSpider.pd_columns[4]],
                )
                for index, row in data.iterrows()
            ]
            DbshHotel.objects.bulk_create(instances, ignore_conflicts=True)
            logger.info("Hotel data inserted into database")
    except Exception as e:
        logger.error(f"An error occurred in the hotel spider task: {str(e)}")


@app.task
def auto_hkvisitors_spider():
    logger.info("Starting the HK visitors spider task")
    try:
        spider = HKVisitorsSpider()
        spider.run()
        data = spider.data()
        max_date_db = DbHkVisitorsImputed.objects.aggregate(max_date=Max('date'))['max_date']
        max_date_spd = data['date'].min()
        if max_date_spd >= max_date_db:
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
                for index, row in data.iterrows()
            ]
            DbHkVisitorsImputed.objects.bulk_create(instances, ignore_conflicts=True)
            logger.info("HK visitors data updated in database")
    except Exception as e:
        logger.error(f"An error occurred in the HK visitors spider task: {str(e)}")


@app.task
def latest_model(directory):
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


@app.task
def autotrain():
    logger.info("Starting auto-training task")
    try:
        base_dir = os.path.join(settings.BASE_DIR, 'models')
        model_dir = latest_model(base_dir)
        db_data = DbShvisitorsDaily.objects.all()
        dataset = melt(cut(read_frame(db_data)))
        train(model_dir, dataset)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred in the auto-training task: {str(e)}")


@app.task
def autopredict():
    logger.info("Starting auto-prediction task")
    try:
        base_dir = os.path.join(settings.BASE_DIR, 'models')
        model_dir = latest_model(base_dir)
        db_data = DbShvisitorsDaily.objects.all()
        dataset = melt(cut(read_frame(db_data)))
        pred = predict(model_dir, dataset, 'AutoPatchTST')
        instances = [
            DbShvisitorsDailyPredicted(
                date=row['date'],
                FOREIGN=row['global_entry'],
                HM=row['hkmo_entry'],
                TW=row['tw_entry']
            )
            for index, row in pred.iterrows()
        ]
        DbHkVisitorsImputed.objects.bulk_create(instances, update_conflicts=True)
        logger.info("Prediction data updated in database")
    except Exception as e:
        logger.error(f"An error occurred in the auto-prediction task: {str(e)}")


@app.task
def pipeline():
    logger.info("Starting pipeline task")
    try:
        fetch_data = group(auto_hotel_spider, auto_hkvisitors_spider)
        renew_model = autopredict | autotrain
        workflow = fetch_data | renew_model
        workflow.apply_async()
        logger.info("Pipeline task execution triggered")
    except Exception as e:
        logger.error(f"An error occurred in the pipeline task: {str(e)}")
