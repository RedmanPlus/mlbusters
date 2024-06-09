FROM python:3.10-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y locales && \
    apt-get install -y python-dev-is-python3 libldap2-dev libsasl2-dev libssl-dev librdkafka-dev netcat-openbsd && \
    echo ru_RU.UTF-8 UTF-8 >> /etc/locale.gen && \
    locale-gen && \
    python -m pip install poetry && \
    poetry config virtualenvs.create false
COPY pyproject.toml poetry.lock ./
RUN python -m pip install --upgrade pip && poetry install --no-root --no-cache
COPY ./ /app/

EXPOSE 8000
CMD python app.py
