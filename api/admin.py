from django.contrib import admin
from api.models import DbshHotel, DbShvisitorsBycountry, DbShvisitorsMonthly

# Register your models here.
admin.site.register(DbshHotel)
admin.site.register(DbShvisitorsBycountry)
admin.site.register(DbShvisitorsMonthly)
