from functools import cached_property
from hashlib import md5
from pathlib import Path
from shutil import rmtree

from django.conf import settings
# pylint: disable=imported-auth-user
from django.contrib.auth.models import User
from django.db import models
from django.db.models import signals as si
from django.dispatch import receiver

from web.ops.passes import Passes

# Create your models here.

# pylint: disable=unused-argument


class API(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    key = models.CharField(max_length=32)

    @staticmethod
    def get_key(data: str) -> str:
        return md5((data + settings.SECRET_KEY).encode('utf-8')).hexdigest()


@receiver(si.post_save, sender=User)
def create_user_API(sender, instance, created, **kwargs):
    if created:
        API.objects.create(user=instance)


@receiver(si.post_save, sender=User)
def save_user_API(sender, instance, **kwargs):
    instance.api.key = API.get_key(instance.username)
    instance.api.save()


class Task(models.Model):
    SEP = '|'
    id: int
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=80)
    hash = models.CharField(max_length=32)
    f_name = models.CharField(max_length=80)
    f_sign = models.CharField(max_length=500)
    path = models.FilePathField()
    date = models.DateTimeField(auto_now_add=True)
    tests = models.PositiveSmallIntegerField()
    ready = models.BooleanField(default=False)
    additional_ops_args = models.CharField(max_length=500)
    cpuinfo = models.CharField(max_length=80)
    _compilers = models.CharField(max_length=32)
    _cflags = models.CharField(max_length=32)
    _passes = models.CharField(max_length=32)

    WORKER_TYPE = [
        ('default', 'Стандартный воркер'),
        ('gpu', 'Более быстрый воркер'),
        ('highmem', 'Стандартный воркер с бОльшим объёмом памяти')
    ]

    _worker_types = models.CharField(max_length=100, default='default')

    @property
    def worker_types(self):
        return list(self._worker_types.split(self.SEP))

    @worker_types.setter
    def worker_types(self, data: list[str]):
        if not isinstance(data, list):
            raise ValueError
        self._worker_types = self.SEP.join(data)

    @property
    def compilers(self):
        return list(map(int, self._compilers.split(self.SEP)))

    @compilers.setter
    def compilers(self, data: list[str]):
        if not isinstance(data, list):
            raise ValueError
        self._compilers = self.SEP.join(map(str, data))

    @property
    def cflags(self):
        return list(self._cflags.split(self.SEP))

    @cflags.setter
    def cflags(self, data: list[str]):
        if not isinstance(data, list):
            raise ValueError
        self._cflags = self.SEP.join(data)

    @property
    def passes(self):
        return list(map(int, self._passes.split(self.SEP)))

    @passes.setter
    def passes(self, data: list[str]):
        if not isinstance(data, list):
            raise ValueError
        self._passes = self.SEP.join(map(str, data))

    def mkdir(self) -> None:
        self.path = settings.TASKS_PATH / str(self.id)
        self.path.mkdir()

    def rmdir(self) -> None:
        rmtree(self.path)


class Result(models.Model):
    id: int
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    num = models.PositiveIntegerField(null=True)
    error = models.BooleanField(default=False)
    worker_type = models.CharField(max_length=10, default='default')
    return_code = models.IntegerField(null=True)

    @cached_property
    def path(self):
        return Path(self.task.path) / f'{self.num}_{self.worker_type}/{self.task.hash}.{Passes(self.num)}_{self.worker_type}.tar.gz'

    @cached_property
    def text(self):
        return Passes(self.num).desc


class Benchmark(models.Model):
    id: int
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    pas = models.PositiveSmallIntegerField(null=True)
    value = models.FloatField(null=True)
    unit = models.CharField(max_length=4)
    error = models.BooleanField(default=False)
    compiler = models.PositiveSmallIntegerField(null=True)
    cflags = models.CharField(max_length=4)
    worker_type = models.CharField(max_length=10, default='default')
    return_code = models.IntegerField(null=True)


class CompError(models.Model):
    id: int
    bench = models.ForeignKey(Benchmark, on_delete=models.CASCADE)
    text = models.TextField(null=True)
