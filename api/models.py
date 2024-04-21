from django.db import models


class DbShvisitorsMonthly(models.Model):
    DATE = models.DateTimeField(primary_key=True, db_column='date')
    FOREIGN = models.IntegerField(db_column='global_entry')
    HM = models.IntegerField(db_column='hkmo_entry')
    TW = models.IntegerField(db_column='tw_entry')

    class Meta:
        db_table = 'sh_visitors'


class DbShvisitorsDaily(models.Model):
    DATE = models.DateTimeField(primary_key=True, db_column='date')
    FOREIGN = models.IntegerField(db_column='global_entry')
    HM = models.IntegerField(db_column='hkmo_entry')
    TW = models.IntegerField(db_column='tw_entry')

    class Meta:
        db_table = 'sh_visitors_daily'


class DbShvisitorsDailyPredicted(models.Model):
    DATE = models.DateTimeField(primary_key=True, db_column='date')
    FOREIGN = models.IntegerField(db_column='global_entry')
    HM = models.IntegerField(db_column='hkmo_entry')
    TW = models.IntegerField(db_column='tw_entry')

    class Meta:
        db_table = 'sh_visitors_daily_pred'


class DbShvisitorsBycountry(models.Model):
    date = models.DateTimeField(primary_key=True, db_column='date')
    country = models.CharField(max_length=100)
    month_visits = models.IntegerField()

    class Meta:
        db_table = 'sh_visitors_bycountry'


class DbshHotel(models.Model):
    DATE = models.DateTimeField(primary_key=True, db_column='日期')
    avg_rent_rate = models.FloatField(db_column='平均出租率')
    avg_price = models.IntegerField(db_column='平均房价')
    avg_rent_rate_5 = models.FloatField(db_column='五星级平均出租率')
    avg_price_5 = models.IntegerField(db_column='五星级平均房价')

    class Meta:
        db_table = 'sh_hotel'


class DbHkVisitorsImputed(models.Model):
    date = models.DateTimeField(primary_key=True)
    HK_airport_entry = models.FloatField(null=True, blank=True, db_column='HK_airport_entry')
    CN_airport_entry = models.FloatField(null=True, blank=True, db_column='CN_airport_entry')
    global_airport_entry = models.FloatField(null=True, blank=True, db_column='global_airport_entry')
    airport_entry = models.FloatField(null=True, blank=True, db_column='airport_entry')
    HK_airport_departure = models.FloatField(null=True, blank=True, db_column='HK_airport_departure')
    CN_airport_departure = models.FloatField(null=True, blank=True, db_column='CN_airport_departure')
    global_airport_departure = models.FloatField(null=True, blank=True, db_column='global_airport_departure')
    airport_departure = models.FloatField(null=True, blank=True, db_column='airport_departure')

    class Meta:
        db_table = 'hk_visitors_imputed'


class DbSpider(models.Model):
    unique_id = models.CharField(max_length=100, db_column='unique_id',primary_key=True)
    url = models.CharField(max_length=1000, db_column='url')

    class Meta:
        db_table = 'spd_tasks'
