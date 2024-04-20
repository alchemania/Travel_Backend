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
    path("api/data/sh/visitors/all", api.views.api_sh_visitors_all),

    path('api/data/sh/visitors/sum/<str:freq>/<int:year>/<int:month>/<int:day>', api.views.api_sh_visitors_sum,
         name='api-sh-visitors-sum'),

    path('api/data/sh/visitors/yoy/<str:freq>/<int:year>/<int:month>/<int:day>', api.views.api_sh_visitors_yoy,
         name='api-sh-visitors-yoy'),

    path("api/data/hotel/all", api.views.api_hotel_all),
    path("api/data/hotel/per", api.views.api_hotel_rate),

    path("api/data/country/rate", api.views.api_country_rate),

    path("ml/train/re", ml.views.ml_re_train),
    path("ml/pred/re", ml.views.ml_re_pred),
    path("ml/pred/is", ml.views.ml_forecast_insight),
    path("ml/paras", ml.views.ml_adjust_paras),
    path("ml/all", ml.views.ml_get_all),

    path('', TemplateView.as_view(template_name='index.html'))
]
