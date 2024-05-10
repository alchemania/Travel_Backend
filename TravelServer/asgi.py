"""
ASGI config for TravelServer project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os

from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "TravelServer.settings")

application = get_asgi_application()
application = ASGIStaticFilesHandler(application)
import socketio
from api.views import ws

application = socketio.ASGIApp(ws, application)

# Django can only handle ASGI/HTTP connections, not lifespan.
# uvicorn TravelServer.asgi:application --host 127.0.0.1 --port 8000
