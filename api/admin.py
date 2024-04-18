from django.contrib import admin
from api.models import DbshHotel, DbShvisitorsBycountry, DbShvisitors

# Register your models here.
admin.site.register(DbshHotel)
admin.site.register(DbShvisitorsBycountry)
admin.site.register(DbShvisitors)
