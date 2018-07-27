FROM bluelens/tensorflow:latest-gpu-py3


RUN mkdir /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

#RUN curl https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl -o ./tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
#RUN pip3 install tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH $PYTHONPATH:/usr/src/app

CMD ["python", "grpc/object_detect_server.py"]
