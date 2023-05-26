import datetime
import os
import django
import keras
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TravelServer.settings')
django.setup()

import numpy as np
from celery import Celery
from dateutil.relativedelta import relativedelta

from api.models import NonMainlandTravelData, HotelData
from ml.models import Ml
from django_pandas.io import read_frame
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import datasets, layers, optimizers, Sequential, metrics

broker = 'redis://127.0.0.1:6379'
backend = 'redis://127.0.0.1:6379/0'

app = Celery('my_task', broker=broker, backend=backend)


@app.task
def re_train(mid: int):
    specModel = Ml.objects.get(id=mid)
    src = specModel.dataSource
    if src == NonMainlandTravelData._meta.db_table:
        # 超参数
        DATA_SCALE = 3
        TIME_STEP = 4
        PRED_STEP = 1
        PRED_LENGTH = specModel.forecastSpan
        LEARINING_RATE = 0.1
        datas = NonMainlandTravelData.objects.filter(DATE__gte=specModel.dataLearnedBegin,
                                                     DATE__lte=specModel.dataLearnedEnd)
        inp = read_frame(qs=datas)
        inp = inp.drop(columns=["DATE", "SUM"]).values
        # 数据标准化
        scr = MinMaxScaler()
        std = scr.fit_transform(inp)
        # 输入数据规范化
        train_x, train_y = [], []
        for i in range(TIME_STEP, std.shape[0] - PRED_STEP + 1):
            train_x.append(std[i - TIME_STEP:i])
            train_y.append(std[i])
        train_x, train_y = np.array(train_x), np.array(train_y)
        # 构建全局网络，包含卷积层，池化层，双向LSTM层
        network = Sequential([
            layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='sigmoid'),
                                 input_shape=(TIME_STEP, DATA_SCALE)),
            layers.Flatten(),
            layers.Dense(DATA_SCALE, activation='sigmoid')
        ])
        # 编译网络
        network.compile(optimizer=optimizers.Adam(learning_rate=LEARINING_RATE),
                        loss=tf.losses.binary_crossentropy,
                        metrics=['accuracy']
                        )
        # 训练网络
        network.fit(train_x, train_y, epochs=100, batch_size=1)
        # save and evaluate
        network.save(f'./models/{specModel.id}.h5')
        flag = network.evaluate(train_x, train_y, verbose=0)
        specModel.loss = flag[0]
        specModel.precision = flag[1]
        # 单步多次预测
        specModel.save()
    elif src == HotelData._meta.db_table:
        pass


'''
有时候，模型训练完成之后也许不需要马上更新数据
也可以之后再调用re_pred
'''


@app.task
def re_pred(mid: int):
    specModel = Ml.objects.get(id=mid)
    src = specModel.dataSource
    if src == NonMainlandTravelData._meta.db_table:
        datas = NonMainlandTravelData.objects.filter(DATE__gte=specModel.dataLearnedBegin,
                                                     DATE__lte=specModel.dataLearnedEnd)
        inp = read_frame(qs=datas)
        inp = inp.drop(columns=["DATE", "SUM"]).values
        # 数据标准化
        scr = MinMaxScaler()
        std = scr.fit_transform(inp)
        # 超参数
        DATA_SCALE = 3
        TIME_STEP = 4
        PRED_LENGTH = specModel.forecastSpan
        network = keras.models.load_model(f'./models/{specModel.id}.h5')
        out = []
        step = std[-TIME_STEP:]
        for _ in range(PRED_LENGTH):
            tmp = network.predict(step.reshape(-1, TIME_STEP, DATA_SCALE))
            step = np.append(step, tmp, axis=0)
            step = np.delete(step, 0, 0)
            out.append(scr.inverse_transform(tmp).reshape(DATA_SCALE))
        out = np.array(out)
        val = np.sum(out, axis=1).reshape(PRED_LENGTH, -1)  # 合计项
        # 生成时间轴
        START_TIME = specModel.dataLearnedEnd
        timeline = [(START_TIME + relativedelta(months=i + 1)) for i in range(PRED_LENGTH)]
        res = np.concatenate([val, out], axis=1).astype(int)
        # res = pd.DataFrame(np.concatenate([timeline, val, out], axis=1), columns=["DATE", "SUM", "FOREIGN", "HM", "TW"])
        # 删除大于dataEnd的数据，是预测值，没有意义
        print(timeline)
        NonMainlandTravelData.objects.filter(DATE__gt=specModel.dataLearnedEnd).delete()
        pred = [NonMainlandTravelData(
            DATE=timeline[index],
            SUM=x[0],
            FOREIGN=x[1],
            HM=x[2],
            TW=x[3]
        ) for index, x in enumerate(res)]
        NonMainlandTravelData.objects.bulk_create(pred)
