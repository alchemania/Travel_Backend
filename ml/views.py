from tasks import re_train, re_pred

from django.http import JsonResponse


def ml_re_train(request):
    id = request.POST["id"]
    re_train.delay((id))
    return JsonResponse({'status': 'training', 'id': id})


def ml_re_pred(request):
    id = request.POST["id"]
    re_pred.delay((id))
    return JsonResponse({'status': 'preding', 'id': id})
