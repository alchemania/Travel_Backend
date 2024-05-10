"""
ASGI config for TravelServer project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""
import json
import os
import shlex
import uuid

from asgiref.sync import sync_to_async
from django.contrib.auth import authenticate
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TravelServer.settings")

application = get_asgi_application()

import socketio

ws = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
application = socketio.ASGIApp(ws, application)


def wsSucessResponse(msg):
    resp = {
        'status': 'success',
        'msg': msg
    }
    return json.dumps(resp)


def wsFailureResponse(msg):
    resp = {
        'status': 'error',
        'msg': msg
    }
    return json.dumps(resp)


@ws.event
def ping(sid, msg):
    return wsSucessResponse("pong!")


@ws.event
def task(sid, msg):
    print("sid:", sid, "msg", msg)
    return wsSucessResponse(msg)


@ws.event
@sync_to_async
def auth(sid, msg: str):
    args = shlex.split(msg)[1:]
    if len(args) != 2:
        return wsFailureResponse("Format Error: auth usr psw")
    usr, psw = args
    user = authenticate(username=usr, password=psw)
    if user is None:
        return wsFailureResponse("Authentication failed")
    return wsSucessResponse(str(uuid.uuid4()))

# Django can only handle ASGI/HTTP connections, not lifespan.
# uvicorn TravelServer.asgi:application --host 127.0.0.1 --port 8000
