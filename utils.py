from PIL import Image
import numpy as np
import tensorflow as tf

# colour map
label_colours = [(0,0,0),(0,0,64),(64,64,0),(64,0,64),(0,64,64)
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                ,(128,0,64),(0,128,64),(0,0,192),(0,192,128),(128,0,192)
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                ,(0,128,192),(64,64,128),(64,128,64),(128,64,64),(192,192,128)
                ,(192,192,64),(192,128,192),(128,128,192),(128,64,128),(192,64,192)]


def decode_labels(mask, num_images=1, num_classes=40):
    n, h, w, c = mask.shape
    assert(n >= num_images), "Batch size %d should be greater or equal than number of images to save %d." % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_,j_] = label_colours[k]
                else:
                    pixels[k_,j_] = (255,255,255)
        outputs[i] = np.array(img)
    return outputs

def inv_preprocess(imgs, num_images, img_mean):
    n, h, w, c = imgs.shape
    assert(n >= num_images), "Batch size %d should be greater or equal than number of images to save %d." % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i] + img_mean)[:, :, ::-1].astype(np.uint8)
    return outputs
