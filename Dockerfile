FROM bluelens/tensorflow:latest-gpu-py3

RUN mkdir -p /opt/app/model

RUN curl https://s3.ap-northeast-2.amazonaws.com/bluelens-style-model/object_detection/label_map.pbtxt -o /opt/app/model/label_map.pbtxt
RUN curl https://s3.ap-northeast-2.amazonaws.com/bluelens-style-model/object_detection/label_map.txt -o /opt/app/model/label_map.txt
RUN curl https://s3.ap-northeast-2.amazonaws.com/bluelens-style-model/object_detection/frozen_inference_graph.pb -o /opt/app/model/frozen_inference_graph.pb
#RUN curl https://s3.ap-northeast-2.amazonaws.com/bluelens-style-model/classification/inception_v3/classify_image_graph_def.pb -o /opt/app/model/classify_image_graph_def.pb

ENV OD_MODEL=/opt/app/model/frozen_inference_graph.pb
ENV OD_LABELS=/opt/app/model/label_map.pbtxt
#ENV CLASSIFY_GRAPH=/opt/app/model/classify_image_graph_def.pb

RUN mkdir /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 50052

WORKDIR /usr/src/app/grpc

CMD ["python", "object_detect_server.py"]
