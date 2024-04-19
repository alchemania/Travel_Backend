from django.db import models


class DbShvisitors(models.Model):
    DATE = models.DateField(primary_key=True, db_column='date')
    FOREIGN = models.IntegerField(db_column='global_entry')
    HM = models.IntegerField(db_column='hkmo_entry')
    TW = models.IntegerField(db_column='tw_entry')

    class Meta:
        db_table = 'sh_visitors'


class DbShvisitorsDaily(models.Model):
    DATE = models.DateField(primary_key=True, db_column='date')
    FOREIGN = models.IntegerField(db_column='global_entry')
    HM = models.IntegerField(db_column='hkmo_entry')
    TW = models.IntegerField(db_column='tw_entry')

    class Meta:
        db_table = 'sh_visitors_daily'


class DbShvisitorsBycountry(models.Model):
    date = models.DateField(primary_key=True, db_column='date')
    country = models.CharField(max_length=100)
    month_visits = models.IntegerField()

    class Meta:
        db_table = 'sh_visitors_bycountry'


class DbshHotel(models.Model):
    DATE = models.DateField(primary_key=True, db_column='日期')
    avg_rent_rate = models.FloatField(db_column='平均出租率')
    avg_price = models.IntegerField(db_column='平均房价')
    avg_rent_rate_5 = models.FloatField(db_column='五星级平均出租率')
    avg_price_5 = models.IntegerField(db_column='五星级平均房价')

    class Meta:
        db_table = 'sh_hotel'
