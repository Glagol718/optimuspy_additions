import shutil
import subprocess as sp
import tarfile
from os import chdir, getcwd
from pathlib import Path

import re

import msgpack
from celery.utils.log import get_logger
from cpuinfo import get_cpu_info

from optimuspy import celery_app
from web.models import Benchmark, CompError, Result, Task
from web.ops.build_tools import catch2
from web.ops.compilers import Compiler, Compilers
from web.ops.passes import Pass, Passes

from celery import group, chain, chord
from itertools import product
from celery.result import allow_join_result
import os
import uuid
import time
from uuid import uuid4

from concurrent.futures import ThreadPoolExecutor
from django.db import close_old_connections
from django.db import transaction

logger = get_logger(__name__)
# channel_layer = get_channel_layer()

# pylint: disable=broad-exception-caught

def get_task_queue(task_id):
    from web.models import Task
    return Task.objects.get(id=task_id).task_type


def publish_message_to_group(message, group: str) -> None:
    with celery_app.producer_pool.acquire(block=True) as producer:

        #не было раньше!
        producer.channel.exchange_declare(
            exchange='groups',
            type='direct',
            durable=True,
            auto_delete=False
        )

        #было раньше!
        producer.publish(
            msgpack.packb({
                "__asgi_group__": group,
                **message,
            }),
            exchange="groups",  # groups_exchange
            content_encoding="binary",
            routing_key=group,
            retry=False,  # Channel Layer at-most once semantics
        )


@celery_app.task
def compiler_job_default(task_id: int):
    return compiler_job(task_id, 'default')

@celery_app.task
def compiler_job_gpu(task_id: int):
    return compiler_job(task_id, 'gpu')

@celery_app.task
def compiler_job_highmem(task_id: int):
    return compiler_job(task_id, 'highmem')

@celery_app.task
def run_single_benchmark_default(task_id: int, pass_num: int, compiler_id: int, cflag_name: str, return_code: int):
    return run_single_benchmark(task_id, pass_num, compiler_id, cflag_name, return_code, 'default')

@celery_app.task
def run_single_benchmark_gpu(task_id: int, pass_num: int, compiler_id: int, cflag_name: str, return_code: int):
    return run_single_benchmark(task_id, pass_num, compiler_id, cflag_name, return_code, 'gpu')

@celery_app.task
def run_single_benchmark_highmem(task_id: int, pass_num: int, compiler_id: int, cflag_name: str, return_code: int):
    return run_single_benchmark(task_id, pass_num, compiler_id, cflag_name, return_code, 'highmem')


@celery_app.task
def compiler_job(task_id: int, worker_type: str):
    try:
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            logger.info('Task does not exist')
            return
        
        path = Path(task.path)
        logger.info(f'Compiler job path to task is {path}')
        files = list(path.iterdir())
        ret_codes = []
        for pass_num in task.passes:
            r = Result(task=task, num=pass_num, worker_type=worker_type)
            r.save()
            
            subdir = path / f"{pass_num}_{worker_type}"
            subdir.mkdir(exist_ok=True)
            
            for file in files:
                if file.is_file():
                    shutil.copy(file, subdir)

            _p = Passes(pass_num)
            p = _p.obj(subdir.iterdir())
            if task.additional_ops_args:
                p.args = task.additional_ops_args.split() + p.args
                logger.info(f"Args are {p.args}")
            
            _ret = p.run()
            if _ret != 0 and _ret != 2:
                r.error = True
                r.save()
            logger.info(f'Return code is {str(_ret)}')
            r.return_code = _ret
            ret_codes.append(_ret)
            cwd = getcwd()
            try:
                chdir(subdir)
                files2 = list(Path('.').iterdir())
                with tarfile.open(f'{task.hash}.{Passes(pass_num)}_{worker_type}.tar.gz', 'w:gz') as tar:
                    for file in files2:
                        tar.add(file)

                for file in files2:
                    if file.is_file():
                        file.unlink()
                    else:
                        shutil.rmtree(file)
            except Exception as e3:
                logger.error(f"Archiving error: {e3}")
                r.error = True
                r.save()
            finally:
                chdir(cwd)
            
            tasks = [
                run_single_benchmark.s(
                    task_id, 
                    pass_num, 
                    comp_id, 
                    cf_name, 
                    _ret,
                    worker_type
                ).set(queue=worker_type)
                for comp_id in task.compilers
                for cf_name in task.cflags
                if any(cf.name == cf_name for cf in Compilers(comp_id).obj.cflags)
            ]
            
            group(tasks).apply_async()

        task.cpuinfo = get_cpu_info()['brand_raw']
        task.ready = True
        task.save()

    except Exception as e:
        logger.error(f"Compiler job failed: {e}")
        task.ready = True
        task.save()
        raise
    return ret_codes

@celery_app.task
def run_single_benchmark(task_id: int, pass_num: int, compiler_id: int, cflag_name: str, return_code: int, worker_type: str):
    task = Task.objects.get(id=task_id)
    path = Path(task.path)
    
    subdir = path / f"{pass_num}_{compiler_id}_{cflag_name}_{worker_type}"
    subdir.mkdir(exist_ok=True)
    
    for file in path.iterdir():
        if file.is_file():
            shutil.copy(file, subdir)

    comp = Compilers(compiler_id).obj
    cf = comp.cflags[cflag_name]
    
    b = Benchmark(
        task=task, 
        pas=pass_num, 
        compiler=compiler_id, 
        cflags=cf.name,
        worker_type=worker_type,
        return_code=return_code
    )
    b.save()
    logger.info(f"Benchmark ret is {return_code}")

    try:
        catch2.setup(subdir, [f.name for f in subdir.iterdir() if f.name.endswith('.c')], task, comp, cf)
        
        cwd = getcwd()
        try:
            chdir(subdir)
            ps1 = sp.run(['make', 'build'], check=False, capture_output=True)
            
            if ps1.returncode != 0:
                CompError.objects.create(bench=b, text=ps1.stderr.decode('utf-8'))

            ps2 = sp.run(['make', 'test'], check=False)
        except Exception as e2:
            logger.error(f"Benchmark failed: {e2}")
            b.error = True
            b.save()
        finally:
            chdir(cwd)
            
        v, u = catch2.parse_benchmark(subdir)
        b.value = v
        b.unit = u
        b.error = (u == 'err')
        b.save()
        files = [f for f in path.iterdir()]
        catch2.cleanup(subdir, files)
        
    except Exception as e3:
        logger.error(f"Benchmark processing error: {e3}")
        b.error = True
        b.save()
        raise
