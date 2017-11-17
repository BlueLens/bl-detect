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

"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import grpc

from PIL import Image
import tensorflow as tf

import object_detect_pb2
import object_detect_pb2_grpc



def run():
  channel = grpc.insecure_channel('localhost:50052')
  stub = object_detect_pb2_grpc.DetectStub(channel)

  file = '/Users/bok95/Desktop/img2.jpg'
  im = Image.open(file)
  size = 300, 300
  im.thumbnail(size, Image.ANTIALIAS)
  im.save('img.jpg')
  with tf.gfile.GFile('img.jpg', 'rb') as fid:
    image_data = fid.read()

  objects = stub.GetObjects(object_detect_pb2.DetectRequest(file_data=image_data))

  for object in objects:
    print(object)

if __name__ == '__main__':
  run()
