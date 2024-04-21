import re

import eventlet

import pipeline.model

eventlet.monkey_patch()

import datetime
import json
import os
import django
import torch
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TravelServer.settings')
django.setup()

import numpy as np
from django_pandas.io import read_frame

from celery import Celery

# django imports
from api.models import *
from django.db.models import Max
from TravelServer import settings

from pipeline.spider import HKVisitorsSpider, SHHotelSpider
from pipeline.model import train, predict
from pipeline.dataprocess import melt, pivot, cut

broker = 'redis://127.0.0.1:6379'
backend = 'redis://127.0.0.1:6379/0'

app = Celery('my_task', broker=broker, backend=backend)

from celery.schedules import crontab

app.conf.beat_schedule = {
    'spider_every_5_minutes': {
        'task': 'tasks.execute_task',
        'schedule': crontab(minute='*/5')
    }
}


@app.task
def auto_hotel_spider():
    tasks = DbSpider.objects.filter(unique_id__contains=SHHotelSpider.identifier).all()
    urls = [[task.unique_id, task.url] for task in tasks]
    spider = SHHotelSpider('update', tasks=urls)
    spider.run()
    # 如果有新任务，更新tasklist
    if spider.tasks:
        objs = [DbSpider(unique_id=uid, url=url) for uid, url in spider.tasks]
        DbSpider.objects.bulk_create(objs, ignore_conflict=True)
        # 接着更新data, 读取为dataframe格式
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
        DbshHotel.objects.bulk_create(instances, ignore_conflict=True)


@app.task
def auto_hkvisitors_spider():
    spider = HKVisitorsSpider()
    spider.run()
    data = spider.data()
    # 如果有新任务，更新tasklist
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
        DbHkVisitorsImputed.objects.bulk_create(instances, ignore_conflict=True)


@app.task
def latest_modelgroup(directory):
    max_num = -1
    max_folder = None
    pattern = re.compile(r'modelgroup_(\d+)')

    # 遍历目录中的所有项
    for item in os.listdir(directory):
        # 构建完整的文件路径
        full_path = os.path.join(directory, item)
        # 检查是否为目录
        if os.path.isdir(full_path):
            # 使用正则表达式检查文件夹名符合 'modelgroup_xxxxxx' 的格式
            match = pattern.match(item)
            if match:
                # 提取数字部分，并转换为整数
                num = int(match.group(1))
                # 检查这个数字是否是最大的
                if num > max_num:
                    max_num = num
                    max_folder = full_path
    return max_folder


@app.task
def train():
    base_dir = os.path.join(settings.BASE_DIR, 'models')
    model_dir = latest_modelgroup(base_dir)
    db_data = DbShvisitorsDaily.objects.all()
    dataset = melt(cut(read_frame(db_data)))
    pipeline.model.train(model_dir, dataset)


@app.task
def predict():
    base_dir = os.path.join(settings.BASE_DIR, 'models')
    model_dir = latest_modelgroup(base_dir)
    db_data = DbShvisitorsDaily.objects.all()
    dataset = melt(cut(read_frame(db_data)))
    pred = pipeline.model.predict(model_dir, dataset, 'AutoPatchTST')

