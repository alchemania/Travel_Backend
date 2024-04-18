import datetime
import json
import os
import django
import keras
import pandas as pd

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TravelServer.settings')
django.setup()

import numpy as np
from celery import Celery
from dateutil.relativedelta import relativedelta

from api.models import DbShvisitors, DbshHotel
from ml.models import Ml
from django_pandas.io import read_frame
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import datasets, layers, optimizers, Sequential

broker = 'redis://127.0.0.1:6379'
backend = 'redis://127.0.0.1:6379/0'

app = Celery('my_task', broker=broker, backend=backend)


@app.task
def re_train(mid: int):
    specModel = Ml.objects.get(id=mid)
    paras = json.loads(specModel.hyperParameters)
    # 超参数 begin
    DATA_SCALE = paras['DATA_SCALE']
    TIME_STEP = paras['TIME_STEP']
    PRED_STEP = paras['PRED_STEP']
    PRED_LENGTH = paras['PRED_LENGTH']
    LEARNING_RATE = paras['LEARNING_RATE']
    LSTM_LAYER = paras['LSTM_LAYER']
    EPOCHS = paras['EPOCHS']
    BATCH_SIZE = paras['BATCH_SIZE']
    # 超参数 end
    if specModel.dataSource == DbShvisitors._meta.db_table:
        data = DbShvisitors.objects.filter(DATE__gte=specModel.dataLearnedBegin,
                                           DATE__lte=specModel.dataLearnedEnd)
        inp = read_frame(qs=data)
        inp = inp.drop(columns=["DATE", "SUM"]).values
    elif specModel.dataSource == DbshHotel._meta.db_table:
        data = DbshHotel.objects.filter(DATE__gte=specModel.dataLearnedBegin,
                                        DATE__lte=specModel.dataLearnedEnd)
        inp = read_frame(qs=data)
        inp = inp.drop(columns=["DATE"]).values
    else:
        return

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
        layers.Bidirectional(layers.LSTM(LSTM_LAYER, return_sequences=True, activation='sigmoid'),
                             input_shape=(TIME_STEP, DATA_SCALE)),
        layers.Flatten(),
        layers.Dense(DATA_SCALE, activation='sigmoid')
    ])
    # 编译网络
    network.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=tf.losses.binary_crossentropy,
                    metrics=['accuracy']
                    )
    # 训练网络
    network.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # save and evaluate
    network.save(f'./models/{specModel.id}.h5')
    flag = network.evaluate(train_x, train_y, verbose=0)
    print(flag)
    # save status
    specModel.loss = flag[0]
    specModel.precision = flag[1]
    specModel.lastUpdate = datetime.datetime.now()
    specModel.modelStatus = "latest"
    specModel.save()


'''
有时候，模型训练完成之后也许不需要马上更新数据
也可以之后再调用re_pred
'''


@app.task
def predict(mid: int, isinsight: bool):
    specModel = Ml.objects.get(id=mid)
    src = specModel.dataSource
    paras = json.loads(specModel.hyperParameters)
    # 超参数
    DATA_SCALE = paras['DATA_SCALE']
    TIME_STEP = paras['TIME_STEP']
    PRED_STEP = paras['PRED_STEP']
    PRED_LENGTH = paras['PRED_LENGTH']
    LEARNING_RATE = paras['LEARNING_RATE']
    if src == DbShvisitors._meta.db_table:
        datas = DbShvisitors.objects.filter(DATE__gte=specModel.dataLearnedBegin,
                                            DATE__lte=specModel.dataLearnedEnd)
        inp = read_frame(qs=datas)
        inp = inp.drop(columns=["DATE", "SUM"]).values
        # 数据标准化
        scr = MinMaxScaler()
        std = scr.fit_transform(inp)
        network = keras.models.load_model(f'./models/{specModel.id}.h5')
        out = []
        step = std[-TIME_STEP:]
        for _ in range(PRED_LENGTH):
            tmp = network.predict(step.reshape(-1, TIME_STEP, DATA_SCALE))
            step = np.append(step, tmp, axis=0)
            step = np.delete(step, 0, 0)
            out.append(scr.inverse_transform(tmp).reshape(DATA_SCALE))
        out = np.array(out).astype(int)
        val = np.sum(out, axis=1).reshape(PRED_LENGTH, -1).astype(int)  # 合计项
        # 生成时间轴
        START_TIME = specModel.dataLearnedEnd
        timeline = [(START_TIME + relativedelta(months=i + 1)) for i in range(PRED_LENGTH)]
        # 删除大于dataEnd的数据，是预测值，没有意义
        if isinsight:
            res = np.concatenate([np.array(timeline).reshape(PRED_LENGTH, -1), val, out], axis=1)
            df = pd.DataFrame(res, columns=["DATE", "SUM", "FOREIGN", "HM", "TW"])
            return df.to_dict(orient='records')
        else:
            res = np.concatenate([val, out], axis=1)
            DbShvisitors.objects.filter(DATE__gt=specModel.dataLearnedEnd).delete()
            pred = [DbShvisitors(
                DATE=timeline[index],
                SUM=x[0],
                FOREIGN=x[1],
                HM=x[2],
                TW=x[3]
            ) for index, x in enumerate(res)]
            DbShvisitors.objects.bulk_create(pred)
    elif specModel.dataSource == DbshHotel._meta.db_table:
        datas = DbshHotel.objects.filter(DATE__gte=specModel.dataLearnedBegin,
                                         DATE__lte=specModel.dataLearnedEnd)
        inp = read_frame(qs=datas)
        inp = inp.drop(columns=["DATE"]).values
        # 数据标准化
        scr = MinMaxScaler()
        std = scr.fit_transform(inp)
        network = keras.models.load_model(f'./models/{specModel.id}.h5')
        out = []
        step = std[-TIME_STEP:]
        for _ in range(PRED_LENGTH):
            tmp = network.predict(step.reshape(-1, TIME_STEP, DATA_SCALE))
            step = np.append(step, tmp, axis=0)
            step = np.delete(step, 0, 0)
            out.append(scr.inverse_transform(tmp).reshape(DATA_SCALE))
        out = np.array(out).astype(int)
        # 生成时间轴
        START_TIME = specModel.dataLearnedEnd
        timeline = [(START_TIME + relativedelta(months=i + 1)) for i in range(PRED_LENGTH)]
        if isinsight:
            res = np.concatenate([np.array(timeline).reshape(PRED_LENGTH, -1), out], axis=1)
            df = pd.DataFrame(res, columns=["DATE", "avg_rent_rate", "avg_price", "avg_rent_rate_5", "avg_price_5"])
            return df.to_dict(orient='records')
        # 删除大于dataEnd的数据，是预测值，没有意义
        else:
            DbshHotel.objects.filter(DATE__gt=specModel.dataLearnedEnd).delete()
            pred = [DbshHotel(
                DATE=timeline[index],
                avg_rent_rate=x[0],
                avg_price=x[1],
                avg_rent_rate_5=x[2],
                avg_price_5=x[3]
            ) for index, x in enumerate(out)]
            DbshHotel.objects.bulk_create(pred)
