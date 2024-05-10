"""
WSGI config for TravelServer project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TravelServer.settings")
application = get_wsgi_application()

import socketio
from api.views import ws

# ws = socketio.Server(cors_allowed_origins='*', async_mode='eventlet')
application = socketio.Middleware(ws, application)

import eventlet
import eventlet.wsgi

eventlet.wsgi.server(eventlet.listen(('', 8000)), application)
