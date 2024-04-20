import datetime
import json
import os
import django
import torch
import neuralforecast
import pypots
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TravelServer.settings')
django.setup()

import numpy as np
from celery import Celery
from dateutil.relativedelta import relativedelta

from api.models import DbShvisitorsMonthly, DbshHotel
from ml.models import Ml
from django_pandas.io import read_frame
from sklearn.preprocessing import MinMaxScaler

broker = 'redis://127.0.0.1:6379'
backend = 'redis://127.0.0.1:6379/0'

app = Celery('my_task', broker=broker, backend=backend)


@app.task
def train():
    pass


@app.task
def predict():
    pass
