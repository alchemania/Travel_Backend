import datetime

import requests
from django.db.models import Sum, Min
from django.http import JsonResponse

from api.models import DbShvisitors, DbshHotel, DbShvisitorsBycountry


# 查出nmainland表中所有数据
def api_sh_visitors_all(request):
    data = DbShvisitors.objects.all()
    res = {
        "timeline": [],
        "frn": [],
        "hk_mw": [],
        "tw": []
    }
    # 对每一项数据重命名
    for i in data:
        res['timeline'].append(str(i.DATE))
        res['frn'].append(i.FOREIGN)
        res['hk_mw'].append(i.HM)
        res['tw'].append(i.TW)
    return JsonResponse(res)


# 计算出sh表中某一年的所有入境人数
def api_nmainland_sum_year(request, year):
    try:
        today = datetime.datetime.today()
        # 判断是否为今年，是今年的话数据不完全，返回目前为止的数据
        if year == today.year:
            year_sum = DbShvisitors.objects.filter(DATE__lte=today, DATE__year=year).aggregate(Sum('SUM'))
        else:
            year_sum = DbShvisitors.objects.filter(DATE__year=year).aggregate(Sum('SUM'))
        return JsonResponse({'sum': 0 if year_sum['SUM__sum'] is None else year_sum['SUM__sum']})
    except Exception:
        return JsonResponse({'sum': 0})


# 计算某一年的同比增长
def api_nmainland_per_year(request, year):
    # n_sum = DbShvisitors.objects.filter(DATE__year=year).aggregate(Sum('SUM'))['SUM__sum']
    # l_sum = DbShvisitors.objects.filter(DATE__year=(year - 1)).aggregate(Sum('SUM'))['SUM__sum']

    # 给定年份的开始和结束日期
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)

    # 前一年的开始和结束日期
    prev_start_date = datetime.date(year - 1, 1, 1)
    prev_end_date = datetime.date(year - 1, 12, 31)

    # 计算给定年份的总访客数
    current_year_total = DbShvisitors.objects.filter(
        DATE__range=(start_date, end_date)
    ).aggregate(total=Sum('FOREIGN') + Sum('HM') + Sum('TW'))['total'] or 0

    # 计算前一年的总访客数
    previous_year_total = DbShvisitors.objects.filter(
        DATE__range=(prev_start_date, prev_end_date)
    ).aggregate(total=Sum('FOREIGN') + Sum('HM') + Sum('TW'))['total'] or 0

    # 计算同比增长率
    growth = ((current_year_total - previous_year_total) / previous_year_total) * 100

    return JsonResponse({'per': growth})


# 返回某一个月的入境人数
def api_nmainland_sum_month(request, year, month):
    month_sum = DbShvisitors.objects.filter(DATE__lte=f'{year}-{month}-1').aggregate(closest_date=Min('DATE'))
    return JsonResponse({'sum': month_sum})


# 计算某一个月的同比增长
def api_nmainland_per_month(request, year, month):
    n_sum = DbShvisitors.objects.filter(DATE__lte=f'{year}-{month}-1').last().SUM
    l_sum = DbShvisitors.objects.filter(DATE__lte=f'{year - 1}-{month}-1').last().SUM
    return JsonResponse({'per': round((n_sum - l_sum) / l_sum, 2)})


# 查询酒店的所有数据
def api_hotel_all(request):
    data = DbshHotel.objects.all()
    res = {
        "timeline": [],
        "ap": [],
        "ap5": [],
        "ar": [],
        "ar5": []
    }
    # 改名称
    for i in data:
        res['timeline'].append(str(i.DATE))
        res['ap'].append(i.avg_price)
        res['ap5'].append(i.avg_price_5)
        res['ar'].append(i.avg_rent_rate)
        res['ar5'].append(i.avg_rent_rate_5)
    return JsonResponse(res)


# 其实这个是预测api，但是预测的数据已经放入数据库，所以直接查询
def api_hotel_rate(request):
    today = datetime.datetime.today()
    # rate = DbshHotel.objects.filter(DATE__year=today.year, DATE__month=today.month).first().avg_rent_rate
    return JsonResponse({'per': 31})


# 废弃api
def api_weather(request):
    city_code = '310000'
    key = '66ca50b578c7a66cc0fd79dcb48f096a'
    url = f'https://restapi.amap.com/v3/weather/weatherInfo?city={city_code}&key={key}'
    wdata = requests.get(url=url).json()
    tmp = wdata["lives"][0]
    return JsonResponse({'data': 'this api has been aborted.'})


# 返回国家排名
def api_country_rate(request):
    # 计算每个国家的总入境人数
    total_visits = DbShvisitorsBycountry.objects.values('country').annotate(all_num=Sum('month_visits'))

    # 计算2019年的总入境人数
    visits_2019 = DbShvisitorsBycountry.objects.filter(date__year=2019).values('country').annotate(cur_num=Sum('month_visits'))

    # 计算2018年的总入境人数
    visits_2018 = DbShvisitorsBycountry.objects.filter(date__year=2018).values('country').annotate(prev_num=Sum('month_visits'))

    # 转换查询结果为字典方便后续处理
    d_2019 = {item['country']: item['cur_num'] for item in visits_2019}
    d_2018 = {item['country']: item['prev_num'] for item in visits_2018}
    sum_all = {item['country']: item['all_num'] for item in total_visits}

    # 创建最终的数据结构
    results = []
    for k in sum_all.keys():
        cur_per = ((d_2019.get(k, 0) - d_2018.get(k, 0)) * 100 / d_2018.get(k, 0)) if d_2018.get(k, 0) else 0
        results.append({
            'country': k,
            'all_num': round(sum_all[k] / 10000, 2),
            'cur_num': round(d_2019.get(k, 0) / 10000, 2),
            'cur_per': round(cur_per, 2)
        })

    res = sorted(results, key=lambda x: x['cur_num'], reverse=True)  # 排序并逆序，最高的排最前
    return JsonResponse(res, safe=False)
