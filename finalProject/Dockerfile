# Задаём базовый образ
#FROM tiangolo/uwsgi-nginx-flask:python3.9
FROM python:3.9
# Копируем содержимое папки ./app в рабочую директорию контейнера
COPY ./app ./
# Копируем файл requirements.txt в рабочую директорию контейнера
COPY ./requirements.txt ./
# Копируем файл uwsgi.ini в рабочую директорию контейнера
#COPY ./uwsgi.ini ./
# Запускаем установку необходимых зависимостей
RUN pip install -r requirements.txt
CMD ["python", "./model.py"]