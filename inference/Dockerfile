FROM python:3.10-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY download_whisper.sh /app/
RUN ./download_whisper.sh

RUN apt-get update && apt-get install ffmpeg -y

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY ./ /app/

EXPOSE 8040
CMD uvicorn main:app --port 80 --log-config=log_conf.yaml
