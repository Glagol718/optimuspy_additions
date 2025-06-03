from django.urls import path

from . import consumers

websocket_urlpatterns = [
    path('ws/ready/', consumers.TaskReadyConsumer.as_asgi())
]
