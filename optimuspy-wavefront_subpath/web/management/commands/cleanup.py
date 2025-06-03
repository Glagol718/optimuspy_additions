from os import system

from django.core import management
from django.core.management.base import BaseCommand

from web.models import API, User


class Command(BaseCommand):
    help = 'Clear db, remove task dirs, purge task queue'

    # pylint: broad-exception-caught
    def handle(self, *args, **options):
        try:
            self.stdout.write(self.style.WARNING('Removing database entries...'))
            management.call_command('migrate', 'web', 'zero')

            self.stdout.write(self.style.WARNING('Migrating database...'))
            management.call_command('migrate', 'web')

            self.stdout.write(self.style.WARNING('Restoring API keys...'))
            for u in User.objects.all():
                a = API(user=u)
                a.key = API.get_key(u.username)
                a.save()

            self.stdout.write(self.style.WARNING('Removing task directories...'))
            management.call_command('rmtasks')

            self.stdout.write(self.style.WARNING('Purging task queue...'))
            system('celery purge -f')

            self.stdout.write(self.style.SUCCESS('Done'))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(exc))
