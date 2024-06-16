FROM python:3.10-bullseye

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install ffmpeg -y
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
COPY ./ /app/

EXPOSE 8040
CMD uvicorn clip:app --port 8040
