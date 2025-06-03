from django.core.management.base import BaseCommand
from django.db import connections
from django.db.utils import OperationalError
import time


class Command(BaseCommand):
    help = 'Django command to pause execution until database is available'

    def handle(self, *args, **kwargs):
        self.stdout.write('waiting for db ...')
        db_conn = None
        while not db_conn:
            try:
                # get the database with keyword 'default' from settings.py
                db_conn = connections['default']
                # prints success messge in green
                self.stdout.write(self.style.SUCCESS('Database online'))
            except OperationalError:
                self.stdout.write('Database offline, waiting 1 second ...')
                time.sleep(1)
