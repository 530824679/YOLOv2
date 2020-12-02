import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import tensorflow as tf
from utils.process_utils import *
from model.network import Network
from cfg.config import *

def predict_image():
    image_path = "/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/originalPics/2003/07/11/big/img_116.jpg"
    image = cv2.imread(image_path)

    input_shape = (416, 416)
    image_shape = image.shape[:2]
    image_normal = preprocess(image, input_shape)

    input = tf.placeholder(tf.float32,[1, input_shape[0], input_shape[1], 3])

    network = Network(False)
    logits = network.build_network(input)
    output = network.reorg_layer(logits, np.array(model_params['anchors']) / (32, 32))

    checkpoints = "./checkpoints/model.ckpt-19000"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoints)
        bboxes, obj_probs, class_probs = sess.run(output, feed_dict={input: image_normal})

    bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs, image_shape=image_shape)

    img_detection = visualization(image, bboxes, scores, class_max_index, model_params["classes"])
    cv2.imshow("result", img_detection)
    cv2.waitKey(0)

if __name__ == "__main__":
    predict_image()