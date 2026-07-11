FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mongo-spark-connector_2.12-10.3.0-all.jar /app/jars/
COPY app.py ./

CMD ["python", "-u", "app.py"]
