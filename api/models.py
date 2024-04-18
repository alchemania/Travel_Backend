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
    DATE = models.DateField(primary_key=True, db_column='日期')
    Japan = models.IntegerField(db_column='日本')
    Singapore = models.IntegerField(db_column='新加坡')
    Tailand = models.IntegerField(db_column='泰国')
    Korea = models.IntegerField(db_column='韩国')
    US = models.IntegerField(db_column='美国')
    Canada = models.IntegerField(db_column='加拿大')
    UK = models.IntegerField(db_column='英国')
    Franch = models.IntegerField(db_column='法国')
    German = models.IntegerField(db_column='德国')
    Italy = models.IntegerField(db_column='意大利')
    Russia = models.IntegerField(db_column='俄罗斯')
    Australia = models.IntegerField(db_column='澳大利亚')
    NewZealand = models.IntegerField(db_column='新西兰')

    class Meta:
        db_table = 'sh_visitors_bycountry'


class DbshHotel(models.Model):
    DATE = models.DateField(primary_key=True, db_column='日期')
    avg_rent_rate = models.FloatField(db_column='星级平均出租率')
    avg_price = models.IntegerField(db_column='星级平均房价')
    avg_rent_rate_5 = models.FloatField(db_column='五星级平均出租率')
    avg_price_5 = models.IntegerField(db_column='五星级平均房价')

    class Meta:
        db_table = 'sh_hotel'
