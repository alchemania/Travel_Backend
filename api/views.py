import datetime
from django.db.models import Sum, F, Avg
from django.db.models.functions import ExtractYear, ExtractMonth
from django.http import JsonResponse
from django.views.decorators.cache import cache_page

from api.models import *
import numpy as np
from django_pandas.io import read_frame


@cache_page(timeout=60 * 5)  # l3
def api_sh_visitors_rawdata(request, freq, ys, ms, ds, ye, me, de):
    start_date = datetime.date(ys, ms, ds)
    end_date = datetime.date(ye, me, de)
    if freq == 'm':
        data = DbShvisitorsMonthly.objects.filter(
            DATE__gte=start_date, DATE__lte=end_date
        )
    elif freq == 'd':
        data = DbShvisitorsDailyPredicted.objects.filter(
            DATE__gte=start_date, DATE__lte=end_date
        )
    else:
        return JsonResponse({'error': 'freq must be "m" or "d"'})
    dct = read_frame(data).to_dict(orient='list')
    dct['DATE'] = list(map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'), dct['DATE']))
    return JsonResponse(dct)


@cache_page(timeout=60 * 5)  # l1
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


@cache_page(timeout=60 * 5)  # l1
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


@cache_page(timeout=60 * 5)  # l3
def api_sh_hotel_rawdata(request, freq, ys, ms, ds, ye, me, de):
    start_date = datetime.date(ys, ms, ds)
    end_date = datetime.date(ye, me, de)
    if freq == 'm':
        data = DbShHotel.objects.filter(
            DATE__gte=start_date, DATE__lte=end_date
        )
    elif freq == 'd':
        return JsonResponse({'error': 'No d frequency in this table'})
    else:
        return JsonResponse({'error': 'freq must be "m" or "d"'})
    dct = read_frame(data).to_dict(orient='list')
    dct['DATE'] = list(map(lambda x: datetime.datetime.strftime(x, '%Y-%m-%d'), dct['DATE']))
    return JsonResponse(dct)


@cache_page(timeout=60 * 5)  # l1
# 其实这个是预测api，但是预测的数据已经放入数据库，所以直接查询
def api_sh_hotel_yoy(request, freq, year, month, day):
    today = datetime.datetime.today()
    rate = DbShHotelPred.objects.filter(DATE__year=year, DATE__month=month).first().avg_rent_rate
    return JsonResponse({'per': rate})


@cache_page(timeout=60 * 30)  # l2
def api_sh_visitors_by_country_statistics(request):
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


@cache_page(timeout=60 * 5)  # l1
def api_sh_datastats(request):
    """
    定义今年相较去年热度
    1. 远小于=-2 -30%
    2. 小于=-1 -30%～-10%
    3  略小于
    3. 相差不大=0
    4. 略大于
    5. 大于
    6. 远大于
    """

    def mapper(percentage: float) -> int:
        if percentage < -0.3:
            return -3
        elif -0.3 <= percentage < -0.15:
            return -2
        elif -0.15 <= percentage < 0.05:
            return -1
        elif -0.05 <= percentage < 0.05:
            return 0
        elif 0.05 <= percentage < 0.15:
            return 1
        elif 0.15 <= percentage < 0.3:
            return 2
        elif 0.3 <= percentage:
            return 3

    # 01
    today = datetime.datetime.now()
    percentage_increment = 0.1
    total_visits_cur_y = DbShvisitorsDailyPredicted.objects.filter(
        DATE__gte=datetime.date(today.year, 1, 1),
        DATE__lte=datetime.date(today.year, today.month, today.day)
    ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']

    total_visits_prev_y = DbShvisitorsDailyPredicted.objects.filter(
        DATE__gte=datetime.date(today.year - 1, 1, 1),
        DATE__lte=datetime.date(today.year - 1, today.month, today.day)
    ).aggregate(total=Sum(F('FOREIGN') + F('HM') + F('TW')))['total']

    hot_year = mapper((total_visits_cur_y - total_visits_prev_y) / total_visits_prev_y)

    # 02
    total_visits_monthly = DbShvisitorsDailyPredicted.objects.filter(DATE__year=today.year).annotate(
        year=ExtractYear('DATE'),
        month=ExtractMonth('DATE')
    ).values('year', 'month').annotate(
        total_foreign=Sum('FOREIGN'),
        total_hm=Sum('HM'),
        total_tw=Sum('TW'),
        total_all=Sum(F('FOREIGN') + F('HM') + F('TW'))
    ).order_by('year', 'month')

    total_visits_monthly_dict = read_frame(total_visits_monthly).to_dict(orient='list')

    hot_month = mapper((total_visits_monthly_dict['total_all'][today.month - 1] /
                        np.average(total_visits_monthly_dict['total_all']) - 1))

    # 02
    recent_average = DbShvisitorsDailyPredicted.objects.filter(
        DATE__gte=today.date(),
        DATE__lt=datetime.date(today.year, today.month + 1, today.day)
    ).order_by('-DATE')[:30].aggregate(
        total=Avg(F('FOREIGN') + F('HM') + F('TW'))
    )['total']
    # 定义高峰阈值，即平均值的指定百分比以上
    threshold = recent_average * (1 + percentage_increment)
    # 找出第一个超过这个阈值的日子
    next_peak_day = DbShvisitorsDailyPredicted.objects.filter(
        DATE__gt=today.date(),
    ).values('DATE').annotate(
        total_entry=Sum(F('FOREIGN') + F('HM') + F('TW'))
    ).filter(
        total_entry__gt=threshold
    ).order_by('DATE').first()

    return JsonResponse({
        'peak_day': next_peak_day['DATE'].strftime('%Y-%m-%d'),
        'hot_year': hot_year,
        'hot_month': hot_month
    })
