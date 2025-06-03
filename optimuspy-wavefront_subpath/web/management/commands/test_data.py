from django.core.management.base import BaseCommand

from web.models import User


class Command(BaseCommand):
    help = 'Create or delete test data in production database'

    def create_or_delete_user(self, uname, pwd):
        try:
            User.objects.get(username=uname).delete()
            self.stdout.write(self.style.WARNING(f'Deleted {uname} user'))
        except User.DoesNotExist:
            User.objects.create_user(username=uname, password=pwd)
            self.stdout.write(self.style.SUCCESS(f'Created {uname} user'))

    # pylint: broad-exception-caught

    def handle(self, *args, **options):
        try:
            self.create_or_delete_user('unittest', '123')

        except Exception as exc:
            self.stdout.write(self.style.ERROR(exc))
