import datetime
import json
import time

# from tasks import re_train, predict
from ml.models import Ml

from django.http import JsonResponse
from django.forms.models import model_to_dict


def predict():
    pass


def re_train():
    pass


def ml_re_train(request):
    pReq = json.loads(request.body)
    mid = pReq['id']
    pReq['hyperParameters'] = json.dumps(pReq['hyperParameters'])
    pReq['lastUpdate'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
    Ml.objects.filter(id=mid).update(**pReq)
    # 返回最新状态
    re_train.delay(mid)
    res = []
    datas = Ml.objects.all()
    for data in datas:
        json_data = model_to_dict(data)
        # 由于dm不支持json，手动转换一次
        json_data['hyperParameters'] = json.loads(json_data['hyperParameters'])
        res.append(json_data)
    return JsonResponse(res, safe=False)


def ml_re_pred(request):
    pReq = json.loads(request.body)
    id = pReq["id"]
    predict.delay(id, False)
    return JsonResponse({'status': 'preding', 'id': id})


# {
#     "paras": {
#         "DATA_SCALE": 3,
#         "TIME_STEP": 4,
#         "PRED_STEP": 1,
#         "LEARINING_RATE": 0.1,
#         "PRED_LENGTH": 12,
#         "PRED_SCALE": "m",
#         "LSTM_LAYER" 32
#     }
# }
# 废弃api
def ml_adjust_paras(request):
    paras = json.loads(request.body)
    ml = Ml.objects.get(id=paras['id'])
    ml.hyperParameters = json.dumps(paras['paras'])
    ml.save()
    return JsonResponse(paras)


def ml_get_all(request):
    res = []
    datas = Ml.objects.all()
    for data in datas:
        json_data = model_to_dict(data)
        # 由于dm不支持json，手动转换一次
        json_data['hyperParameters'] = json.loads(json_data['hyperParameters'])
        res.append(json_data)
    return JsonResponse(res, safe=False)


def ml_forecast_insight(request):
    pReq = json.loads(request.body)
    id = str(pReq['id'])
    res = predict(id, True)
    return JsonResponse({'id': id, 'insight': res})
