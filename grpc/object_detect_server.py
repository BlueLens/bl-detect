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
import time

import grpc

from object_detect import ObjectDetect
import object_detect_pb2
import object_detect_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Detect(object_detect_pb2_grpc.DetectServicer):
  def __init__(self):
    self.od = ObjectDetect()

  def GetObjects(self, request, context):
    # print(request)
    objects = self.od.detect(request.file_data)

    for object in objects:
      detectReply = object_detect_pb2.DetectReply()
      detectReply.class_code = object['class_code']
      detectReply.class_name = object['class_name']
      detectReply.location.left = object['box'][0]
      detectReply.location.right = object['box'][1]
      detectReply.location.top = object['box'][2]
      detectReply.location.bottom = object['box'][3]
      detectReply.feature = object['feature']
      yield  detectReply

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=50))
  object_detect_pb2_grpc.add_DetectServicer_to_server(Detect(), server)
  server.add_insecure_port('[::]:50052')
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  serve()
