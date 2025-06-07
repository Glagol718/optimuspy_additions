from time import sleep
import requests
import base64 as b64
import tarfile
from io import BytesIO



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



class TestingService(object):
    def __init__(self, url, user, password, settings):
        self.url = url
        self.user = user
        self.password = password
        self.token = None
        self.settings = settings
        self.getToken()

    def getToken(self):
        data = {
            'username': self.user,
            'password': self.password,
        }
        r = requests.post(self.url+'/api/auth/', json=data, timeout=60)
        json = r.json()
        self.token = json['token']

    def sendOPSTask(self, extra_arguments, program_text):
        data = {
            'token': self.token,
            'compilers': [self.settings.compiler],
            'passes': [self.settings.pass_name],
            'cflags': [self.settings.cflags],
            'files': make_tar_b64(program_text),
            'additional_ops_args': self.settings.additional_ops_args + extra_arguments + self.settings.after_ops_args,
            'tests': self.settings.iterations,
            'worker_types': [self.settings.worker_types],
        }
        print("sending request")
        print(data)

        r = requests.post(self.url+'/api/submit/', json=data, timeout=60)
        r.raise_for_status()
        print(r.status_code)
        json = r.json()
        task_hash = json["task"]
        print('Task hash is ', task_hash)
        return task_hash

    def waitTaskFinish(self, task_hash):
        data = {
            'token': self.token,
            'task': task_hash
        }
        i = 0
        while True:
            if (i := i + 1) > 15:
                break
            try:
                r = requests.post(self.url+'/api/result/', json=data, timeout=60)
                json = r.json()
                if json['status'] == 'success':
                    return True
            except requests.Timeout:
                return False
            sleep(5)

    def isTaskReady(self, task_hash):
        data = {
            'token': self.token,
            'task': task_hash
        }
        r = requests.post(self.url + '/api/result/', json=data, timeout=60)
        json = r.json()
        if json['status'] == 'success':
            return True
        else:
            return False

    def getTaskRunTime(self, task_hash):
        data = {
            'token': self.token,
            'task': task_hash
        }
        r = requests.post(self.url + '/api/result/', json=data, timeout=60)
        json = r.json()
        if "benchmarks" not in json or not json["benchmarks"]:
            return (-1, -1)
        if not json["benchmarks"][0]["error"]:
            runtime = json["benchmarks"][0]["value"]
            unit = json["benchmarks"][0]["unit"]
        else:
            runtime = -1
        ret_code = json["benchmarks"][0]["return_code"]
        return (runtime, ret_code, unit)
