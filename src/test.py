# desc:    Script to feed a given image through a given model
#          and save the output
# -------------------------------------------------------------
import tensorflow as tf
from keras import backend as K
import numpy as np
from PIL import Image
import model_unet
import util
import sys
import os


if len(sys.argv) != 4:
    print('Usage: python test.py [model_PATH] [in_PATH] [out_dir]')
    exit()

_, model_PATH, in_PATH, out_DIR = sys.argv

IMAGE_SZ = 128

mask = util.load_mask('mask.png')

img = np.array(Image.open(in_PATH).convert('RGB'))
if img.shape != (IMAGE_SZ, IMAGE_SZ, 3):
    print(f'Invalid image size: {list(img.shape)}')
    exit()

img = img[np.newaxis] / 255.0
img_p = util.preprocess_images_inpainting(img, mask=mask)

tf.reset_default_graph()
K.set_learning_phase(1)
G_Z = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 4], name='G_Z')
G_sample = model_unet.generator(G_Z)
K.set_learning_phase(0)

saver = tf.train.Saver()

with tf.Session() as sess:
    K.set_session(sess)
    saver.restore(sess, model_PATH)
    output, = sess.run([G_sample], feed_dict={G_Z: img_p})

    name, _ = os.path.splitext(os.path.basename(in_PATH))

    img_out = output[0]

    # save output image
    out_path = os.path.join(out_DIR, f'{name}_output.png')
    util.save_image(img_out, out_path)

    # save pasted and blended versions
    util.postprocess_images_inpainting(in_PATH, out_path, os.path.join(out_DIR, f'{name}_paste.png'), blend=False, mask=mask)
    util.postprocess_images_inpainting(in_PATH, out_path, os.path.join(out_DIR, f'{name}_blend.png'), blend=True, mask=mask)

