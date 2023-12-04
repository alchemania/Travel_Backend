
# Travel Backend README

## 1. Overview

This project is a django+vue3 data dashboard. This Django backend provides data support for the frontend dashboard, including but not limited to using API calls to specified data, calling predicted data, and querying processed data.

## 2. API Introduction

All APIs of this project can be seen in the URLs in the TravelServer folder.

Below are all the APIs:

```Python
path("admin/", admin.site.urls),
path("api/data/nmainland/all", api.views.api_nmainland_all),
path('api/data/nmainland/sum/<int:year>', api.views.api_nmainland_sum_year),
path('api/data/nmainland/per/<int:year>', api.views.api_nmainland_per_year),
path('api/data/nmainland/sum/<int:year>/<int:month>', api.views.api_nmainland_sum_month),
path('api/data/nmainland/per/<int:year>/<int:month>', api.views.api_nmainland_per_month),
path("api/data/hotel/all", api.views.api_hotel_all),
path("api/data/hotel/per", api.views.api_hotel_rate),
path("api/data/weather", api.views.api_weather),
path("api/data/country/rate", api.views.api_country_rate),
path('', TemplateView.as_view(template_name='index.html'))
```

### Other Data Type APIs

All subsequent APIs follow one principle: API/(return value type, data, img, video, etc.)/ database table/operation / additional conditions/additional conditions...

For example, this API indicates that it seeks to return data type data, needs to query the mainland table, and returns all data.

The operators include all, per, sum, etc., which respectively mean query all, query year-over-year growth, and query total.

The return format is as follows (all APIs return JSON):

All 'all' APIs return all database data, so the format is consistent with the database.

All 'per' return { 'per': num}, num = original value * 100.

All 'sum' return { 'sum': num}, num is generally the original value, in special cases return original value / 10000.

**PS: <...> in the API is a wildcard, formatted as <data type: variable name>**

celery startup command for the win, not needed for Linux
celery worker -A tasks --loglevel=info -P eventlet
celery -A tasks worker --loglevel=info -P eventlet

redis
./redis-server.exe redis.windows.conf

## Required Packages
django==3.1.7
dmPython
django_dmPython
eventlet ==latest
celery == latest
pandas
redis
django-pandas
django-cors-headers
numpy (bundled installation)
scikit_learn
