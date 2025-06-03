from os import system

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Start Celery'

    # pylint: broad-exception-caught
    def handle(self, *args, **options):
        try:
            self.stdout.write(self.style.SUCCESS('Starting celery...'))
            system('celery -A optimuspy worker --concurrency 1 -P solo --loglevel=info')
        except Exception as exc:
            self.stdout.write(self.style.ERROR(exc))
