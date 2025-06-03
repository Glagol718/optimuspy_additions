from shutil import rmtree

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Remove all task directories'

    # pylint: broad-exception-caught
    def handle(self, *args, **options):
        try:
            for p in settings.TASKS_PATH.iterdir():
                self.stdout.write(self.style.NOTICE(f'Deleting {p}...'))
                rmtree(p)
            self.stdout.write(self.style.SUCCESS('Done'))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(exc))
