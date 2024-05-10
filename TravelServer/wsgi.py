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

from api.views import ws
import socketio

application = socketio.WSGIApp(ws, application)
from gevent import pywsgi

pywsgi.WSGIServer(('', 8000), application).serve_forever()
