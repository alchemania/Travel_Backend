import datetime

import requests
from django.db.models import Sum, Min, F
from django.http import JsonResponse

from api.models import DbShvisitorsMonthly, DbShHotel, DbShvisitorsBycountry, DbShvisitorsDaily, \
    DbShvisitorsDailyPredicted
from tasks import *


def api_sh_visitors_all(request):
    data = DbShvisitorsMonthly.objects.all()
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


def api_sh_visitors_sum(request, freq, year, month, day):
    if freq == 'y':
        current_date = datetime.date(year, 1, 1)
        current_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=current_date.year,
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']
    elif freq == 'm':
        if month is None:
            raise ValueError("Month is required for monthly growth calculation")
        current_date = datetime.date(year, int(month), 1)
        current_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=current_date.year,
            DATE__month=current_date.month,
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']
    elif freq == 'd':
        if month is None or day is None:
            raise ValueError("Month and day are required for daily growth calculation")
        current_date = datetime.date(year, month, day)
        current_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=current_date.year,
            DATE__month=current_date.month,
            DATE__day=current_date.day
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']
    else:
        return JsonResponse({'error': 'Frequency must be "y", "m", or "d'"})"})
    return JsonResponse({'sum': current_total})


def api_sh_visitors_yoy(request, freq, year, month, day):
    # 构建当前日期和前一年的日期
    if freq == 'y':
        current_date = datetime.date(year, 1, 1)
        previous_date = datetime.date(year - 1, 1, 1)
        # 计算当前日期的总访问者
        current_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=current_date.year,
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']

        # 计算前一年相同日期的总访问者
        previous_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=previous_date.year,
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']
    elif freq == 'm':
        if month is None:
            raise ValueError("Month is required for monthly growth calculation")
        current_date = datetime.date(year, month, 1)
        previous_date = datetime.date(year - 1, month, 1)
        # 计算当前日期的总访问者
        current_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=current_date.year,
            DATE__month=current_date.month,
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']

        # 计算前一年相同日期的总访问者
        previous_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=previous_date.year,
            DATE__month=previous_date.month,
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']
    elif freq == 'd':
        if month is None or day is None:
            raise ValueError("Month and day are required for daily growth calculation")
        current_date = datetime.date(year, month, day)
        previous_date = datetime.date(year - 1, month, day)
        # 计算当前日期的总访问者
        current_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=current_date.year,
            DATE__month=current_date.month,
            DATE__day=current_date.day
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']

        # 计算前一年相同日期的总访问者
        previous_total = DbShvisitorsDailyPredicted.objects.filter(
            DATE__year=previous_date.year,
            DATE__month=previous_date.month,
            DATE__day=previous_date.day
        ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']
    else:
        raise ValueError("Frequency must be 'y', 'm', or 'd'")

    # 如果没有前一年的数据，则返回None
    if previous_total is None or current_total is None:
        return JsonResponse({'per': 0})

    # 计算同比增长率
    growth = ((current_total - previous_total) / previous_total) * 100 if previous_total != 0 else float('inf')
    return JsonResponse({'per': round(growth, 2)})


# 查询酒店的所有数据
def api_hotel_all(request):
    data = DbShHotel.objects.all()
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
    visits_2019 = DbShvisitorsBycountry.objects.filter(date__year=2019).values('country').annotate(
        cur_num=Sum('month_visits'))

    # 计算2018年的总入境人数
    visits_2018 = DbShvisitorsBycountry.objects.filter(date__year=2018).values('country').annotate(
        prev_num=Sum('month_visits'))

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


def api_maintain_trigger(request, module):
    activity = {
        "parallel_spiders": auto_parallel_spiders.delay(),
        "hkvisitors": auto_hkvisitors_spider.delay(),
        "hotel": auto_hotel_spider.delay(),
        "shvisitors": auto_shvisitors_spider.delay(),
        "train": autopredict.delay(),
        "predict": autopredict.delay(),
        "model_renewal": auto_model_renewal.delay(),
    }

    if module in activity:
        return JsonResponse({'status': 'success', 'msg': f'{module} triggered successfully'})
    else:
        return JsonResponse({'status': 'failed', 'msg': f'{module} triggered failed'})
