FROM python:3.11.9-bookworm

WORKDIR /app

COPY . /app


RUN apt-get update && pip install -r requirements.txt

CMD ["python3", "app.py"]