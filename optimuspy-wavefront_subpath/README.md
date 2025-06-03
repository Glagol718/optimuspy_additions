![](static/images/logo.png)
# Сервис оптимизации программ для выбора лучших цепочек преобразований компилятором

## Описание
Проблема автоматической оптимизации произвольных программ как никогда актуальна в наши дни.

Решением этой проблемы является создание системы, способной автоматически преобразовывать исходные тексты программ для выполнения высокоуровневых оптимизаций, направленных на распараллеливание и оптимизацию работы кэш-памяти в приложении.

Одной из существующих систем такого типа является Оптимизирующая [Распараллеливающая Система (OPS)](https://ops.rsu.ru/en/about.shtml).

Затем возникает потребность в предоставлении легкого доступа к системе широкому кругу пользователей. Для удовлетворения этой потребности и был создан веб-сервис **optimus.py**.

### [Полный текст работы](https://hub.sfedu.ru/repository/material/801314243/)

## Используемые технологии
- [OPS](https://ops.rsu.ru/en/about.shtml)
- [Docker](https://www.docker.com)
- [Django](https://www.djangoproject.com)
- [Celery](https://www.djangoproject.com)
- [PostgreSQL](https://www.postgresql.org)
- [Uvicorn](https://www.uvicorn.org)
- [Nginx](https://nginx.org/ru/)
- [Bootstrap](https://getbootstrap.com)
- [Bokeh](http://bokeh.org)
- [ctags](https://ctags.sourceforge.net)
- [Catch2](https://github.com/catchorg/Catch2)
- [GNU Make](https://www.gnu.org/software/make/)
- [GCC](https://gcc.gnu.org) и [Clang](https://clang.llvm.org)
- [и другие](requirements/)

## Структура базы данных
![](https://raw.githubusercontent.com/COOLIRON2311/opsc-bin/master/images/1.png)

## Архитектура и состав модулей
![](https://raw.githubusercontent.com/COOLIRON2311/opsc-bin/master/images/2.png)

## Развертывание
Для развертывания сервера прежде всего необходимо скачать дистрибутив приложения. Это можно сделать при помощи системы контроля версий **git**:
``` sh
git clone https://github.com/COOLIRON2311/optimuspy --recursive
```
Параметр `--recursive` необходим для автоматического скачивания модуля [opsc-bin](https://github.com/COOLIRON2311/opsc-bin), содержащего динамические библиотеки и исполняемые файлы системы OPS, скомпилированные под Linux.

Если не требуется запускать проект в контейнере, то этот параметр можно не указывать.

После скачивания дистрибутива проекта необходимо внести некоторые изменения в его конфигурационные файлы. Во-первых, в корневой директории проекта необходимо создать файл c именем `.env` и содержимым:
``` sh
SECRET_KEY ='**************************************************'
DJANGO_SETTINGS_MODULE='optimuspy.settings'

POSTGRES_DB=optimuspy
POSTGRES_USER=optimuspy
POSTGRES_PASSWORD=************************
DATABASE_HOST=db
DATABASE_PORT=5432

EMAIL_HOST='smtp.mailgun.org'
EMAIL_PORT=587
EMAIL_HOST_USER='**************************************************.mailgun.org'
EMAIL_HOST_PASSWORD='**************************************************'

SOCIAL_AUTH_GITHUB_KEY='********************'
SOCIAL_AUTH_GITHUB_SECRET='****************************************'

SOCIAL_AUTH_GITLAB_KEY='***************************************************************'
SOCIAL_AUTH_GITLAB_SECRET='****************************************************************'
```
Значения полей, содержащих символ «*», необходимо заменить на полученные от провайдеров соответствующих сервисов.

Затем, если не планируется отладка сервиса, то необходимо внести следующие изменения в файл [optimuspy/settings.py](optimuspy/settings.py):
- Поменять значение переменной `DEBUG` с `True` на `False`.
- В список `CSRF_TRUSTED_ORIGINS` добавить текущий используемый сервисом домен. Например: `'http://optimuspy.ru'`.
- При необходимости изменить пути включения заголовочных файлов стандартной библиотеки C. По умолчанию указаны следующие значения:
``` python
INCLUDES = ['-I=/usr/lib/gcc/x86_64-linux-gnu/11/include',			'-I=/usr/local/include',
		'-I=/usr/include/x86_64-linux-gnu',
 		'-I=/usr/include']
```
- Убедиться, что включены все необходимые компиляторы и проходы системы OPS.
``` python
COMPILERS = [
    Compilers.GCC, Compilers.Clang
]
...

OPS_PASSES = [
    Passes.NoOptPass, Passes.OMPPass, Passes.TilingPass
]
...
```
- По умолчанию в базовый образ [cooliron/optimuspy:base](https://hub.docker.com/repository/docker/cooliron/optimuspy/general) включены компиляторы GCC 11.3.0 и Clang 15.0.7:

- Наконец необходимо собрать и запустить сервис следующей командой:
``` sh
docker compose up --build -d
```
Если же планируется отладка сервиса вне контейнера, то необходимо раскомментировать следующую строку в файле [settings.py](optimuspy/settings.py):
``` python
INSTALLED_APPS = [
    # 'daphne',
...
```
И запустить сервер командой `python manage.py runserver`.

## Пример использования
Проведем оптимизацию и тестирование простой программы для умножения матриц с одновременным подсчетом суммы элементов:
``` c
#define N 500

float matmul(const float *a, int m, int n, const float *b, int p, int q, float *r)
{
    if (n != p)
        return 0;
    float s = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < q; j++)
        {
            for (int k = 0; k < p; k++)
                r[i * m + j] += a[i * m + k] * b[k * p + j];
            s += r[i * m + j];
        }
    }
    return s;
}

void optimus()
{
    float a[N * N], b[N * N], c[N * N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // ...
        }
    }
    volatile float r = matmul(a, N, N, b, N, N, c);
}
```
После авторизации в системе одним из доступных способов, пользователь попадает на домашнюю страницу со списком своих отправлений.

Создадим новое отправление, нажав на кнопку «Отправить».

Зададим следующие параметры:

![](https://raw.githubusercontent.com/COOLIRON2311/opsc-bin/master/images/3.png)

После завершения процесса выполнения экспериментов, пользователь попадает на страницу с результатами их проведения.

![](https://raw.githubusercontent.com/COOLIRON2311/opsc-bin/master/images/4.png)

Со сценариями использования сервиса при взаимодействии с интерфейсом прикладного программирования можно ознакомиться в [модульных тестах проекта](web/tests.py) и в [модуле automation](automation).
