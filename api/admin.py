from django.contrib import admin
from api.models import *

admin.site.site_header = "旅游大数据系统管理面板"

# Register your models here.
admin.site.register(DbShHotel)
admin.site.register(DbShvisitorsBycountry)
admin.site.register(DbShvisitorsMonthly)
admin.site.register(DbShvisitorsDailyPredicted)
admin.site.register(DbShvisitorsDaily)
admin.site.register(DbSpider)
admin.site.register(DbHkVisitorsImputed)

