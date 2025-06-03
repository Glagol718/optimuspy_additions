import base64 as b64
import tarfile
from io import BytesIO
from time import sleep

import django
import requests
from django.conf import settings
from django.test import TestCase
from django.test.runner import DiscoverRunner

# Create your tests here.

# pylint: disable=import-outside-toplevel, global-statement

django.setup()


def make_tar_b64(src: str) -> str:
    out = BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as tar:
        data = BytesIO(src.encode('utf-8'))
        info = tarfile.TarInfo('main.c')
        info.size = data.getbuffer().nbytes
        tar.addfile(info, data)
    # with open('test.tar.gz', 'wb') as f:
    #     f.write(out.getvalue())
    return b64.b64encode(out.getbuffer()).decode('utf-8')


TOKEN: str = None
PROG: str = '''
void optimus()
{
	int A[100];
	int B[100];
	int C[100];
	for(int i = 0; i < 100; i++) {
		C[i] = A[i] + B[i];
	}
}
'''
TAR: str = make_tar_b64(PROG)
TASK: str = None


def setUpModule():
    global TOKEN
    from web.models import User
    u: User = None
    try:
        u = User.objects.get(username='unittest')
    except User.DoesNotExist:
        u = User.objects.create_user(username='unittest', password='123')
    finally:
        TOKEN = u.api.key


def tearDownModule():
    from web.models import Task, User
    u = User.objects.get(username='unittest')
    for t in Task.objects.filter(user=u):
        t.rmdir()
    u.delete()


class ProductionDBTestRunner(DiscoverRunner):
    """Custom test runner to do testing on live database :P"""

    def setup_databases(self, **kwargs):
        pass

    def teardown_databases(self, *args, **kwargs):
        pass


class AuthTests(TestCase):
    def test_invalid_params(self):
        data = {
            'username': '123'
        }
        try:
            r = requests.post('http://localhost:8000/api/auth/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
        except requests.Timeout:
            self.fail('timeout')

    def test_user_does_not_exist(self):
        data = {
            'username': 'xXx_lol_i_am_so_random_XD_xXx',
            'password': '123'
        }
        try:
            r = requests.post('http://localhost:8000/api/auth/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'user does not exist')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_password(self):
        data = {
            'username': 'unittest',
            'password': '1234'
        }
        try:
            r = requests.post('http://localhost:8000/api/auth/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid password')
        except requests.Timeout:
            self.fail('timeout')

    def test_success(self):
        data = {
            'username': 'unittest',
            'password': '123'
        }
        try:
            r = requests.post('http://localhost:8000/api/auth/', json=data, timeout=60)
            json = r.json()
            self.assertFalse(json['error'])
            self.assertEqual(json['status'], 'success')
            self.assertEqual(json['token'], TOKEN)
        except requests.Timeout:
            self.fail('timeout')


class CompilersTests(TestCase):
    def test_compilers(self):
        r = requests.post('http://localhost:8000/api/compilers/', timeout=60)
        json = r.json()
        self.assertFalse(json['error'])
        self.assertEqual(json['compilers'], [c.name for c in settings.COMPILERS])


class PassesTests(TestCase):
    def test_passes(self):
        r = requests.post('http://localhost:8000/api/passes/', timeout=60)
        json = r.json()
        self.assertFalse(json['error'])
        self.assertEqual(json['passes'], [p.name for p in settings.OPS_PASSES])


class TasksTests(TestCase):
    def test_invalid_token(self):
        data = {
            'token': '123'
        }
        try:
            r = requests.post('http://localhost:8000/api/tasks/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid token')
            self.assertEqual(json['tasks'], [])
        except requests.Timeout:
            self.fail('timeout')

    def test_success(self):
        data = {
            'token': TOKEN
        }
        try:
            r = requests.post('http://localhost:8000/api/tasks/', json=data, timeout=60)
            json = r.json()
            self.assertFalse(json['error'])
            self.assertEqual(json['status'], 'success')
        except requests.Timeout:
            self.fail('timeout')


class SubmitTests(TestCase):
    def test_invalid_params(self):
        data = {
            'foo': 'bar'
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_token(self):
        data = {
            'token': '123',
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid token')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_compilers_missing(self):
        data = {
            'token': TOKEN,
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_passes_missing(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_cflags_missing(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_files_missing(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_tests_missing(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': '123'
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_compilers_empty(self):
        data = {
            'token': TOKEN,
            'compilers': [],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'compilers list is empty')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_passes_empty(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': [],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'passes list is empty')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_cflags_empty(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': [],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'cflags list is empty')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_compiler(self):
        data = {
            'token': TOKEN,
            'compilers': ['test'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], "invalid compiler 'test'")
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_pass(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['test'],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], "invalid pass 'test'")
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_cflags(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['test'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], "invalid cflags 'test'")
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_files_not_propely_encoded(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': '123',
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'files are not properly encoded')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_numer_of_tests(self):
        def test():
            try:
                r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
                json = r.json()
                self.assertTrue(json['error'])
                self.assertEqual(json['status'], 'invalid number of tests')
                self.assertEqual(json['task'], '')
            except requests.Timeout:
                self.fail('timeout')
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': TAR,
            'tests': 0
        }
        test()

        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': TAR,
            'tests': 101
        }
        test()

    def test_has_main_func(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': make_tar_b64(PROG.replace('optimus', 'main')),
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'files must not contain main function declaration')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_no_func_signature(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': make_tar_b64('int answer = 42;'),
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'could not find any function signatures in files')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_could_not_resolve(self):
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': make_tar_b64('''
            void foo() {}
            void bar() {}
            '''),
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'could not resolve target function signature')
            self.assertEqual(json['task'], '')
        except requests.Timeout:
            self.fail('timeout')


class GeneralUseCase(TestCase):
    def test_01_submit(self):
        global TASK
        data = {
            'token': TOKEN,
            'compilers': ['GCC'],
            'passes': ['NoOptPass'],
            'cflags': ['O1'],
            'files': TAR,
            'tests': 1
        }
        try:
            r = requests.post('http://localhost:8000/api/submit/', json=data, timeout=60)
            json = r.json()
            self.assertFalse(json['error'])
            self.assertEqual(json['status'], 'success')
            self.assertNotEqual(json['task'], '')
            TASK = json['task']
        except requests.Timeout:
            self.fail('timeout')

    def test_02_tasks(self):
        data = {
            'token': TOKEN,
        }
        try:
            r = requests.post('http://localhost:8000/api/tasks/', json=data, timeout=60)
            json = r.json()
            self.assertFalse(json['error'])
            self.assertEqual(json['status'], 'success')
            self.assertEqual(json['tasks'], [TASK])
        except requests.Timeout:
            self.fail('timeout')

    def test_03_result(self):
        data = {
            'token': TOKEN,
            'task': TASK
        }
        i = 0
        while True:
            if (i := i + 1) > 5:
                break
            try:
                r = requests.post('http://localhost:8000/api/result/', json=data, timeout=60)
                json = r.json()
                if json['status'] == 'success':
                    break
            except requests.Timeout:
                self.fail('timeout')
            sleep(5)

        self.assertFalse(json['error'])
        self.assertEqual(json['status'], 'success')
        self.assertNotEqual(json['benchmarks'], {})

    def test_04_download(self):
        data = {
            'token': TOKEN,
            'task': TASK,
            'pas': 'NoOptPass'
        }
        try:
            r = requests.post('http://localhost:8000/api/download/', json=data, timeout=60)
            json = r.json()

            # self.assertFalse(json['error'])
            self.assertEqual(json['status'], 'success')
            self.assertNotEqual(json['file'], '')
        except requests.Timeout:
            self.fail('timeout')


class ResultTests(TestCase):
    def test_invalid_params(self):
        data = {
            'token': TOKEN
        }
        try:
            r = requests.post('http://localhost:8000/api/result/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['benchmarks'], [])
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_token(self):
        data = {
            'token': '123',
            'task': 'test'
        }
        try:
            r = requests.post('http://localhost:8000/api/result/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid token')
            self.assertEqual(json['benchmarks'], [])
        except requests.Timeout:
            self.fail('timeout')

    def test_no_such_task(self):
        data = {
            'token': TOKEN,
            'task': 'test'
        }
        try:
            r = requests.post('http://localhost:8000/api/result/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'no such task')
            self.assertEqual(json['benchmarks'], [])
        except requests.Timeout:
            self.fail('timeout')


class DownloadTests(TestCase):
    def test_invalid_params(self):
        data = {
            'token': TOKEN
        }
        try:
            r = requests.post('http://localhost:8000/api/download/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid parameters')
            self.assertEqual(json['file'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_token(self):
        data = {
            'token': '123',
            'task': 'test',
            'pas': 'NoOptPass'
        }
        try:
            r = requests.post('http://localhost:8000/api/download/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'invalid token')
            self.assertEqual(json['file'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_invalid_pass(self):
        data = {
            'token': TOKEN,
            'task': 'test',
            'pas': 'test'
        }
        try:
            r = requests.post('http://localhost:8000/api/download/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], "invalid pass 'test'")
            self.assertEqual(json['file'], '')
        except requests.Timeout:
            self.fail('timeout')

    def test_no_such_task(self):
        data = {
            'token': TOKEN,
            'task': 'test',
            'pas': 'NoOptPass'
        }
        try:
            r = requests.post('http://localhost:8000/api/download/', json=data, timeout=60)
            json = r.json()
            self.assertTrue(json['error'])
            self.assertEqual(json['status'], 'no such task')
            self.assertEqual(json['file'], '')
        except requests.Timeout:
            self.fail('timeout')


class CPUInfoTests(TestCase):
    def test_cpuinfo(self):
        try:
            r = requests.post('http://localhost:8000/api/cpuinfo/', timeout=60)
            json = r.json()
            self.assertNotEqual(json, {})
            self.assertTrue('brand_raw' in json)
        except requests.Timeout:
            self.fail('timeout')
