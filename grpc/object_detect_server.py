# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent import futures
from multiprocessing import Process
import redis
import time
import os

import grpc

from object_detect_top import TopObjectDetect
from object_detect_bottom import BottomObjectDetect
from object_detect_full import FullObjectDetect
from object_detect_all import AllObjectDetect
from object_detect_top_full import TopFullObjectDetect
import object_detect_pb2
import object_detect_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

OD_GRPC_PORT = os.environ['OD_GRPC_PORT']
REDIS_SERVER = os.environ['REDIS_SERVER']
REDIS_PASSWORD = os.environ['REDIS_PASSWORD']

rconn = redis.StrictRedis(REDIS_SERVER, port=6379, password=REDIS_PASSWORD)

# model = top_full_model / bottom_model
class Detect(object_detect_pb2_grpc.DetectServicer):
  def __init__(self):
    # self.top_od = TopObjectDetect()
    self.bottom_od = BottomObjectDetect()
    # self.full_od = FullObjectDetect()
    # self.all_od = AllObjectDetect()
    self.top_full_od = TopFullObjectDetect()

  def GetObjects(self, request, context):
    # print(request)
    # top_objects = self.top_od.detect(request.file_data)
    bottom_objects = self.bottom_od.detect(request.file_data)
    # full_objects = self.full_od.detect(request.file_data)
    # all_objects = self.all_od.detect(request.file_data)
    top_full_object = self.top_full_od(request.file_data)

    objects = []
    # objects.extend(top_objects)
    objects.extend(bottom_objects)
    # objects.extend(full_objects)
    # objects.extend(all_objects)
    objects.extend(top_full_object)

    for object in objects:
      detectReply = object_detect_pb2.DetectReply()
      detectReply.class_code = object['class_code']
      detectReply.class_name = object['class_name']
      detectReply.score = object['score']
      detectReply.location.left = object['box'][0]
      detectReply.location.right = object['box'][1]
      detectReply.location.top = object['box'][2]
      detectReply.location.bottom = object['box'][3]
      detectReply.feature = object['feature']
      yield  detectReply

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=500))
  object_detect_pb2_grpc.add_DetectServicer_to_server(Detect(), server)
  server.add_insecure_port('[::]:' + OD_GRPC_PORT)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

def restart(rconn, pids):
  while True:
    key, value = rconn.blpop([REDIS_INDEX_RESTART_QUEUE])
    for pid in pids:
      os.kill(pid, signal.SIGTERM)
    sys.exit()

if __name__ == '__main__':
  serve()
  pids = []
  p1 = Process(target=serve, args=(rconn,))
  p1.start()
  pids.append(p1.pid)
  Process(target=restart, args=(rconn, pids)).start()
