FROM python:3.11-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt /app/


RUN apt-get update && \
    apt-get install -y locales && \
    apt-get install -y python-dev-is-python3 libldap2-dev libsasl2-dev libssl-dev librdkafka-dev netcat-openbsd && \
    sudo apt-get install -y sqlite3 libsqlite3-dev && \
    echo ru_RU.UTF-8 UTF-8 >> /etc/locale.gen && \
    locale-gen

RUN python -m pip install --upgrade pip && pip install -r requirements.txt
RUN python -c "import sqlite3; import time; print(sqlite3.sqlite_version); time.sleep(10)"

COPY ./ /app/

EXPOSE 8000
CMD uvicorn main:app --workers 4 --port 8000
