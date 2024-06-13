FROM python:3.11-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt /app/

# setting up sqlite for chroma connection
RUN wget https://www.sqlite.org/src/tarball/sqlite.tar.gz?r=release \
    -O sqlite.tar.gz && \
    tar xzf sqlite.tar.gz && \
    cd sqlite/ && \
    ./configure && \
    make sqlite3.c &&\
    cp sqlite/sqlite3.[ch] pysqlite3/ && \
    cd pysqlite3 && \
    python setup.py build_static build

RUN apt-get update && \
    apt-get install -y locales && \
    apt-get install -y python-dev-is-python3 libldap2-dev libsasl2-dev libssl-dev librdkafka-dev netcat-openbsd && \
    echo ru_RU.UTF-8 UTF-8 >> /etc/locale.gen && \
    locale-gen
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY ./ /app/

EXPOSE 8000
CMD uvicorn main:app --workers 4 --port 8000
