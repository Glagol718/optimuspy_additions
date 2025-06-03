import webbrowser
from os import system

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Start monitor and administration tool for Celery'

    # pylint: broad-exception-caught
    def handle(self, *args, **options):
        try:
            self.stdout.write(self.style.SUCCESS('Starting flower...'))
            webbrowser.open('http://localhost:5555')
            system('celery -A optimuspy flower')
        except Exception as exc:
            self.stdout.write(self.style.ERROR(exc))
