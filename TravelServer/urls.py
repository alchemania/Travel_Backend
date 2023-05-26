"""TravelServer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from django.views.generic import TemplateView

import api.views
import ml.views

urlpatterns = [
    path("admin", admin.site.urls),
    path("api/data/nmainland/all", api.views.api_nmainland_all),
    path('api/data/nmainland/sum/<int:year>', api.views.api_nmainland_sum_year),
    path('api/data/nmainland/per/<int:year>', api.views.api_nmainland_per_year),
    path('api/data/nmainland/sum/<int:year>/<int:month>', api.views.api_nmainland_sum_month),
    path('api/data/nmainland/per/<int:year>/<int:month>', api.views.api_nmainland_per_month),
    path("api/data/hotel/all", api.views.api_hotel_all),
    path("api/data/hotel/per", api.views.api_hotel_rate),
    path("api/data/weather", api.views.api_weather),
    path("api/data/country/rate", api.views.api_country_rate),

    path("ml/retrain", ml.views.ml_re_train),
    path("ml/repred", ml.views.ml_re_pred),

    path('', TemplateView.as_view(template_name='index.html'))
]
