FROM python:3.11-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt /app/


RUN apt-get update && \
    apt-get install -y locales && \
    echo ru_RU.UTF-8 UTF-8 >> /etc/locale.gen && \
    locale-gen

RUN wget https://www.sqlite.org/src/tarball/sqlite.tar.gz?r=release \
    -O sqlite.tar.gz && \
    tar xzf sqlite.tar.gz && \
    cd sqlite/ && \
    ./configure && \
    make sqlite3.c && \
    apt-get purge sqlite3 && \
    make install && \
    export PATH="/usr/local/bin:$PATH" && \
    cd .libs && \
    cp libsqlite3.so.0 /usr/lib/x86_64-linux-gnu/libsqlite3.so.0

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY ./ /app/

EXPOSE 8080
CMD uvicorn main:app --workers 4 --port 8080
