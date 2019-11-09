#!/usr/bin/env python

import json
import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.client import timeline
import cv2

PATH = os.path.dirname(__file__)
MODEL_FILENAME = "frozen_model.pb"
CATEGORIES_FILENAME = "categories.json"
INPUT_SIZE = (70, 70)
INPUT_TENSOR_NAME = "input_1:0"
OUTPUT_TENSOR_NAME = "fc5/Softmax:0"


class Model(object):

    def __init__(self):
        self.graph = tf.Graph()
        self.graph_def = None

        with open(os.path.join(PATH, MODEL_FILENAME), 'rb') as f:
            self.graph_def = tf.GraphDef.FromString(f.read())

        if self.graph_def is None:
            print('Error loading graph')

        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=self.config)
        self.run_options = tf.RunOptions()
        self.run_metadata = tf.RunMetadata()

    @property
    def trace(self):
        return self.run_options.trace_level > 0

    @trace.setter
    def trace(self, value):
        if value:
            self.run_options.trace_level = tf.RunOptions.FULL_TRACE
        else:
            self.run_options.trace_level = 0

    def infer(self, images):
        new_images = [cv2.resize(image, INPUT_SIZE) for image in images]

        cls_scores = self.session.run(
            OUTPUT_TENSOR_NAME,
            options=self.run_options,
            run_metadata=self.run_metadata,
            feed_dict={INPUT_TENSOR_NAME: new_images}
        )

        if self.run_options.trace_level > 0:
            fetched_timeline = timeline.Timeline(self.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            filename = "/tmp/trace-" + str(time.time()) + ".json"
            with open(filename, "w") as f:
                f.write(chrome_trace)

        return cls_scores
