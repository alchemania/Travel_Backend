from django.contrib import admin
from api.models import HotelData, ForeignTravelData, NonMainlandTravelData

# Register your models here.
admin.site.register(HotelData)
admin.site.register(ForeignTravelData)
admin.site.register(NonMainlandTravelData)
