# optimuspy_additions
Важно! Для корректной работы следует скачать .env файл и архивы ОРС по ссылке:
https://drive.google.com/drive/folders/1ZHKcpzHgWqv57HJIU8MLxpUMI6G1Cu0F?usp=sharing
После чего необходимо вставить все файлы кроме .env файла в подкаталог opsc-bin корневой папки проекта, и вставить файл .env в саму корневую папку проекта
Затем нужно поднять docker-контейнеры командами:
docker-compose -f docker-compose.yml up -d
docker-compose -f docker-compose-worker.yml up -d
После этого по адресу localhost:8173 можно будет подключиться к проекту
