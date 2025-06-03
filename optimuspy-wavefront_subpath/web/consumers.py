# from asgiref.sync import async_to_sync
from channels.generic.websocket import AsyncWebsocketConsumer


class TaskReadyConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        """Connect channel 'self.channel_name' to 'ready' group"""
        await self.channel_layer.group_add(
            'ready', self.channel_name
        )
        # Accept connection
        await self.accept()

    async def disconnect(self, code):
        """Disconnect channel 'self.channel_name' from 'ready' group"""
        await self.channel_layer.group_discard(
            'ready', self.channel_name
        )

    async def ready_announce(self, event):
        """ready.announce event handler"""
        await self.send(text_data=event['task'])
