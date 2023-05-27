import json

from tasks import re_train, re_pred
from ml.models import Ml

from django.http import JsonResponse
from django.forms.models import model_to_dict


def ml_re_train(request):
    id = request.POST["id"]
    re_train.delay(id)
    return JsonResponse({'status': 'training', 'id': id})


def ml_re_pred(request):
    id = request.POST["id"]
    re_pred.delay(id)
    return JsonResponse({'status': 'preding', 'id': id})


# {
#     "paras": {
#         "DATA_SCALE": 3,
#         "TIME_STEP": 4,
#         "PRED_STEP": 1,
#         "LEARINING_RATE": 0.1,
#         "PRED_LENGTH": 12,
#         "PRED_SCALE": "m"
#     }
# }
def ml_adjust_paras(request):
    paras = json.loads(request.body)
    print(paras)
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
