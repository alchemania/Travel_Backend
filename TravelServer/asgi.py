"""
ASGI config for TravelServer project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os

import uvicorn
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TravelServer.settings")

application = get_asgi_application()

import socketio

ws = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
application = socketio.ASGIApp(ws, application)


@ws.event
def connect(sid, data):
    print('connect')


@ws.event
def cmd(sid, msg):
    print("sid:", sid, "msg", msg)

# uvicorn TravelServer.asgi:application --host 127.0.0.1 --port 8000
