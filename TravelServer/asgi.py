"""
ASGI config for TravelServer project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""
# Django can only handle ASGI/HTTP connections, not lifespan.
# uvicorn TravelServer.asgi:application --host 127.0.0.1 --port 8000

import os
import json

import psutil
import socketio
import shlex
import uuid
import datetime

from asgiref.sync import sync_to_async
from django.apps import apps
from django.contrib.auth import authenticate
from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
from django.core.asgi import get_asgi_application
from django.db.models import Sum, Min, Max, F, Avg, ExpressionWrapper

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TravelServer.settings")

application = get_asgi_application()
application = ASGIStaticFilesHandler(application)

ws = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
application = socketio.ASGIApp(ws, application)


def wsSucessResponse(content):
    resp = {
        'type': 'normal',
        'class': 'success',
        'content': content
    }
    return json.dumps(resp)


def wsFailureResponse(content):
    resp = {
        'type': 'normal',
        'class': 'error',
        'content': content
    }
    return json.dumps(resp)


def wsTableResponse(content):
    resp = {
        "type": "table",
        "content": {
            **content
        }
    }
    return json.dumps(resp)


@ws.event
@sync_to_async
def connect(sid, env, tkn):
    usr = tkn.get('usr')
    psw = tkn.get('psw')
    user = authenticate(username=usr, password=psw)
    if not usr or not psw or not user:
        ws.disconnect(sid)
        return False
    return True


@ws.event
def disconnect(sid):
    print('Client disconnected', sid)


@ws.event
def ping(sid, msg):
    return wsSucessResponse("pong!")


@ws.event
@sync_to_async
def inspectdb(sid, msg):
    all_models = apps.get_models()
    table_details = {
        "head": ["Table Name", "Primary Key", "Min/Max Values", "Row Count"],
        "rows": []
    }

    for model in all_models:
        # 获取每个模型的表名和主键字段
        table_name = model._meta.db_table
        pk_field = model._meta.pk.name

        # 确保不是Django内部表
        if not table_name.startswith("django_"):
            # 获取行数
            row_count = model.objects.count()

            # 使用Django ORM获取主键的最小和最大值
            min_max_values = model.objects.aggregate(Min(pk_field), Max(pk_field))
            min_val = min_max_values[f'{pk_field}__min']
            max_val = min_max_values[f'{pk_field}__max']

            # 添加行到JSON结构中
            table_details["rows"].append([
                table_name,
                f"Primary Key: {pk_field}",
                f"MIN({pk_field})={min_val}, MAX({pk_field})={max_val}",
                f"{row_count} rows"
            ])

    return wsTableResponse(table_details)


@ws.event
@sync_to_async
def inspectsys(sid, msg):
    # 收集数据
    cpu_info = {
        "Cores": psutil.cpu_count(logical=False),
        "CPU Usage": f"{psutil.cpu_percent(interval=1, percpu=True)}%"
    }
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()

    # 构建JSON结构
    system_info = {
        "head": ["Metric", "Value", "Description"],
        "rows": [
            ["CPU Cores", cpu_info["Cores"], "Number of physical CPU cores"],
            ["CPU Usage", cpu_info["CPU Usage"], "Percentage of CPU usage"],
            ["Total Memory", f"{memory_info.total / (1024 ** 3):.2f} GB", "Total physical memory"],
            ["Used Memory", f"{memory_info.used / (1024 ** 3):.2f} GB", f"{memory_info.percent}%"],
            ["Disk Total", f"{disk_info.total / (1024 ** 3):.2f} GB", "Total disk space"],
            ["Disk Used", f"{disk_info.used / (1024 ** 3):.2f} GB", f"{disk_info.percent}%"],
            ["Bytes Sent", f"{net_io.bytes_sent / (1024 ** 2):.2f} MB", "Total bytes sent"],
            ["Bytes Received", f"{net_io.bytes_recv / (1024 ** 2):.2f} MB", "Total bytes received"],
            ["System Uptime",
             f"{(datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600:.2f} hours",
             "Time since last reboot"]
        ]
    }
    return wsTableResponse(system_info)


@ws.event
def task(sid, msg):
    print("sid:", sid, "msg", msg)
    return wsSucessResponse(msg)
