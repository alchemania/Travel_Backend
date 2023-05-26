import uuid

from django.db import models

# Create your models here.


class Ml(models.Model):
    # uuid v3 v5 都是值相同id相同，这里给随机值
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)  # xxx模型
    module = models.CharField(max_length=100)  # .pb,.h5,.savedPoint
    driver = models.CharField(max_length=100)  # tensorflow pytorch sklearn
    dataSource = models.CharField(max_length=100)  # table name
    dataLearnedBegin = models.DateField()  # bg
    dataLearnedEnd = models.DateField()  # ed
    forecastSpan = models.IntegerField()  # 12个月
    forecastUnit = models.CharField(max_length=20)
    lastForecast = models.DateTimeField()  # 上次运行时间
    loss = models.FloatField()  # loss
    precision = models.FloatField()  # 准确度
    # training - info 训练中
    # preding - info 预测中
    # latest - success dataLearnedEnd与当前日期相差一个月以内
    # useful - warning dataLearnedEnd与当前日期相差3个月以内
    # outdated - error dataLearnedEnd与当前日期相差3个月以上
    modelStatus = models.CharField(max_length=20)

    class Meta:
        db_table = "ml_mlInfo"
