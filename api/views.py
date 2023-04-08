import datetime
import json
import math

import requests
from django.forms import model_to_dict
from django.http import HttpResponse, JsonResponse
from django.core import serializers
from api.models import NonMainlandTravelData, HotelData, ForeignTravelData
from django.db.models import Avg, Sum
from django.db.models import F


# Create your views here.
def api_nmainland_all(request):
    data = NonMainlandTravelData.objects.all()
    res = {
        "timeline": [],
        "frn": [],
        "hk_mw": [],
        "tw": []
    }
    for i in data:
        res['timeline'].append(str(i.DATE))
        res['frn'].append(i.FOREIGN)
        res['hk_mw'].append(i.HM)
        res['tw'].append(i.TW)
    return JsonResponse(res)


def api_nmainland_sum_year(request, year):
    try:
        today = datetime.datetime.today()
        if year == today.year:
            year_sum = NonMainlandTravelData.objects.filter(DATE__lte=today, DATE__year=year).aggregate(Sum('SUM'))
        else:
            year_sum = NonMainlandTravelData.objects.filter(DATE__year=year).aggregate(Sum('SUM'))
        return JsonResponse({'sum': 0 if year_sum['SUM__sum'] is None else year_sum['SUM__sum']})
    except Exception:
        return JsonResponse({'sum': 0})


def api_nmainland_per_year(request, year):
    n_sum = NonMainlandTravelData.objects.filter(DATE__year=year).aggregate(Sum('SUM'))['SUM__sum']
    l_sum = NonMainlandTravelData.objects.filter(DATE__year=(year - 1)).aggregate(Sum('SUM'))['SUM__sum']
    return JsonResponse({'per': round(100 * (n_sum - l_sum) / l_sum, 2)})


def api_nmainland_sum_month(request, year, month):
    month_sum = NonMainlandTravelData.objects.filter(DATE__lte=f'{year}-{month}-1').last().SUM
    return JsonResponse({'sum': month_sum})


def api_nmainland_per_month(request, year, month):
    n_sum = NonMainlandTravelData.objects.filter(DATE__lte=f'{year}-{month}-1').last().SUM
    l_sum = NonMainlandTravelData.objects.filter(DATE__lte=f'{year - 1}-{month}-1').last().SUM
    return JsonResponse({'per': round((n_sum - l_sum) / l_sum, 2)})


def api_hotel_all(request):
    data = HotelData.objects.all()
    res = {
        "timeline": [],
        "ap": [],
        "ap5": [],
        "ar": [],
        "ar5": []
    }
    for i in data:
        res['timeline'].append(str(i.DATE))
        res['ap'].append(i.avg_price)
        res['ap5'].append(i.avg_price_5)
        res['ar'].append(i.avg_rent_rate)
        res['ar5'].append(i.avg_rent_rate_5)
    return JsonResponse(res)


def api_hotel_rate(request):
    today = datetime.datetime.today()
    rate = HotelData.objects.filter(DATE__year=today.year, DATE__month=today.month).first().avg_rent_rate
    return JsonResponse({'per': rate})


def api_weather(request):
    city_code = '310000'
    key = '66ca50b578c7a66cc0fd79dcb48f096a'
    url = f'https://restapi.amap.com/v3/weather/weatherInfo?city={city_code}&key={key}'
    wdata = requests.get(url=url).json()
    tmp = wdata["lives"][0]
    return JsonResponse({'data': 'this api has been aborted.'})


def api_country_rate(request):
    sum_all = ForeignTravelData.objects.filter(DATE__lte='2019-12-30') \
        .aggregate(Sum('Japan'), Sum('Singapore'), Sum('Tailand'), Sum('Korea'), Sum('US'), Sum('Canada'), Sum('UK'),
                   Sum('Franch'), Sum('German'), Sum('Italy'), Sum('Russia'), Sum('Australia'), Sum('NewZealand'))
    sum_period = ForeignTravelData.objects.filter(DATE__lte='2019-12-30', DATE__gte='2018-12-30') \
        .aggregate(Sum('Japan'), Sum('Singapore'), Sum('Tailand'), Sum('Korea'), Sum('US'), Sum('Canada'), Sum('UK'),
                   Sum('Franch'), Sum('German'), Sum('Italy'), Sum('Russia'), Sum('Australia'), Sum('NewZealand'))
    sum_tmp = ForeignTravelData.objects.filter(DATE__lte='2018-12-30', DATE__gte='2017-12-30') \
        .aggregate(Sum('Japan'), Sum('Singapore'), Sum('Tailand'), Sum('Korea'), Sum('US'), Sum('Canada'), Sum('UK'),
                   Sum('Franch'), Sum('German'), Sum('Italy'), Sum('Russia'), Sum('Australia'), Sum('NewZealand'))
    sum_all, d_2019, d_2018 = dict(sum_all), dict(sum_period), dict(sum_tmp)
    res = []
    for k in d_2019.keys():
        res.append({
            'country': str(k).split('__')[0],
            'all_num': round(sum_all[k] / 10000, 2),
            'cur_num': round(d_2019[k] / 10000, 2),
            'cur_per': round((d_2019[k] - d_2018[k]) * 100 / d_2018[k], 2)
        })
    res = sorted(res, key=lambda x: x['cur_num'], reverse=True)
    return JsonResponse(res, safe=False)
