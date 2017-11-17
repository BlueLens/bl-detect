# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import object_detect_pb2 as object__detect__pb2


class DetectStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetObjects = channel.unary_stream(
        '/objectdetect.Detect/GetObjects',
        request_serializer=object__detect__pb2.DetectRequest.SerializeToString,
        response_deserializer=object__detect__pb2.DetectReply.FromString,
        )


class DetectServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GetObjects(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DetectServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetObjects': grpc.unary_stream_rpc_method_handler(
          servicer.GetObjects,
          request_deserializer=object__detect__pb2.DetectRequest.FromString,
          response_serializer=object__detect__pb2.DetectReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'objectdetect.Detect', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
