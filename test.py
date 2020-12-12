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
    image_path = "/home/chenwei/HDD/Project/datasets/object_detection/FDDB2016/convert/images/2002_07_19_big_img_90.jpg"
    image = cv2.imread(image_path)
    image_size = image.shape[:2]
    input_shape = [model_params['input_height'], model_params['input_width']]
    image_data = pre_process(image, input_shape)
    image_data = image_data[np.newaxis, ...]

    input = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)

    network = Network(is_train=False)
    logits = network.build_network(input)
    output = network.reorg_layer(logits, model_params['anchors'])

    checkpoints = "./checkpoints/model.ckpt-19000"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoints)
        bboxes, obj_probs, class_probs = sess.run(output, feed_dict={input: image_data})

    bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs, image_shape=image_shape)

    img_detection = visualization(image, bboxes, scores, class_max_index, model_params["classes"])
    cv2.imshow("result", img_detection)
    cv2.waitKey(0)

if __name__ == "__main__":
    predict_image()