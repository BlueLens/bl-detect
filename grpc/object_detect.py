# coding: utf-8

from __future__ import absolute_import

import numpy as np
import os
from PIL import Image
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from stylelens_feature import feature_extract

import io
from util import label_map_util

OD_MODEL = os.environ['OD_MODEL']
OD_LABELS = os.environ['OD_LABELS']

NUM_CLASSES = 3

class ObjectDetect(object):
  def __init__(self):
    label_map = label_map_util.load_labelmap(OD_LABELS)
    print(label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    self.__category_index = label_map_util.create_category_index(categories)
    self.__detection_graph = tf.Graph()
    with self.__detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(OD_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

      self.__sess = tf.Session(graph=self.__detection_graph)

    # self.image_feature = feature_extract.ExtractFeature()
    print('_init_ done')

  def detect(self, image_bytes):

    image_data = Image.open(io.BytesIO(image_bytes))
    image_np = self.load_image_into_numpy_array(image_data)

    show_box = True
    out_image, boxes, scores, classes, num_detections = self.detect_objects(image_np, self.__sess, self.__detection_graph, show_box)

    out_boxes = self.take_object(
      out_image,
      np.squeeze(boxes),
      np.squeeze(scores),
      np.squeeze(classes).astype(np.int32))

    print(out_boxes)
    return out_boxes

  def take_object(self, image_np, boxes, scores, classes):
    max_boxes_to_save = 3
    min_score_thresh = 0.1
    taken_boxes = []
    if not max_boxes_to_save:
      max_boxes_to_save = boxes.shape[0]
    for i in range(min(max_boxes_to_save, boxes.shape[0])):
      if scores is None or scores[i] > min_score_thresh:
        print(scores[i])
        if classes[i] in self.__category_index.keys():
          class_name = self.__category_index[classes[i]]['name']
          class_code = self.__category_index[classes[i]]['code']
        else:
          class_name = 'na'
          class_code = 'na'
        print(boxes.shape)
        print(boxes[i])
        print(boxes[i].shape)
        ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())

        use_normalized_coordinates = True
        left, right, top, bottom = self.crop_bounding_box(
          image_np,
          ymin,
          xmin,
          ymax,
          xmax,
          use_normalized_coordinates=use_normalized_coordinates)
        item = {}

        item['box'] = [left, right, top, bottom]
        item['class_name'] = class_name
        item['class_code'] = class_code
        taken_boxes.append(item)
    return taken_boxes

  def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

  def crop_bounding_box(self,
                        image,
                        ymin,
                        xmin,
                        ymax,
                        xmax,
                        use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array).

    Args:
      image: a numpy array with shape [height, width, 3].
      ymin: ymin of bounding box in normalized coordinates (same below).
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      name: classname
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    im_width, im_height = image_pil.size
    if use_normalized_coordinates:
      (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    else:
      (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    # print(image_pil)
    # area = (left, top, left + abs(left-right), top + abs(bottom-top))
    # cropped_img = image_pil.crop(area)
    # cropped_img.save(TMP_CROP_IMG_FILE)
    # img_feature_vec = self.image_feature.extract_feature(TMP_CROP_IMG_FILE)
    # cropped_img.show()
    # id = self.save_to_db(image_info)


    # try:
    #   api_response = self.__search.search_image(file=TMP_CROP_IMG_FILE)
    #   if api_response.code == 0 and api_response.data != None:
    #     res_images = api_response.data.images
    #
    # except ApiException as e:
    #   print("Exception when calling SearchApi->search_image: %s\n" % e)

    # save_image_to_file(image_pil, ymin, xmin, ymax, xmax,
    #                            use_normalized_coordinates)
    # np.copyto(image, np.array(image_pil))
    # return id, res_images, left, right, top, bottom
    # return id, left, right, top, bottom
    return left, right, top, bottom

  def detect_objects(self, image_np, sess, detection_graph, show_box=True):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})

    if show_box:
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        self.__category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=3,
        min_score_thresh=.05,
        line_thickness=8)
    # print(image_np)
    return image_np, boxes, scores, classes, num_detections
