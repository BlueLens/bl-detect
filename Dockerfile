FROM bluelens/bl-detect-base:latest


RUN mkdir /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 50052

WORKDIR /usr/src/app/grpc

CMD ["python", "object_detect_server.py"]
