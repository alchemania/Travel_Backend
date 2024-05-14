# python packages
import os
import re

from celery import Celery, chain, group
from celery.schedules import crontab
from celery.utils.log import get_task_logger
from django_pandas.io import read_frame

import pipeline

# Set logging
logger = get_task_logger(__name__)

# django imports
from api.models import *
from django.db.models import Max
from TravelServer import settings

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
def auto_hotel_spider(*args, **kwargs):
    logger.info('auto_hotel_spider start')
    pipeline.crawl_sh_hotel()
    logger.info('auto_hotel_spider end')


@app.task
def auto_hkvisitors_spider(*args, **kwargs):
    logger.info('auto_hkvisitors_spider start')
    pipeline.crawl_hk_visitors()
    logger.info('auto_hkvisitors_spider finished')


@app.task
def auto_shvisitors_spider(*args, **kwargs):
    logger.info('auto_shvisitors_spider start')
    pipeline.crawl_sh_visitors()
    logger.info('auto_shvisitors_spider finished')


@app.task
def auto_parallel_spiders(*args, **kwargs):
    logger.info("Starting the parallel spider task")
    parallel = group(auto_hotel_spider.s(), auto_hkvisitors_spider.s(), auto_shvisitors_spider.s())
    parallel.apply_async()
    logger.info("Parallel spider task fetching completed successfully")


@app.task
def impute(*args, **kwargs):
    pipeline.impute_hk_visitors()


@app.task()
def interpolate(*args, **kwargs):
    pipeline.interpolate_sh_visitors()


@app.task
def autotrain(*args, **kwargs):
    pipeline.train_sh_visitors()


@app.task
def autopredict(*args, **kwargs):
    pipeline.predict_sh_visitors()


@app.task
def workflow(*args, **kwargs):
    """
    To start the pipeline task
    1. celery -A tasks worker --loglevel=info -P eventlet 启动worker
    2. celery -A tasks beats --loglevel=info 启动定时任务
    3. celery -A tasks flower 启动flower面板
    """
    logger.info("Starting pipeline task")
    try:
        workflow = chain(auto_parallel_spiders.s(), auto_model_renewal.s())
        workflow.apply_async()
        logger.info("Pipeline task execution triggered")
    except Exception as e:
        logger.error(f"An error occurred in the pipeline task: {str(e)}")
