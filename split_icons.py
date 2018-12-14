import os
import sys
import glob
from PIL import Image


def split(source, destination, width, height):
    if os.path.exists(destination):
        if os.listdir(destination) > 0:
            print destination, " already cropped"
            return os.listdir(destination)
    else:
        os.makedirs(destination)

    if not os.path.isfile(source):
        print 'This is not file: ', source
        sys.exit(1)

    w, h = int(width), int(height)
    im = Image.open(source)
    im_w, im_h = im.size
    print 'Image width:%d height:%d  will split into (%d %d) ' % (im_w, im_h, w, h)
    w_num, h_num = int(im_w / w), int(im_h / h)

    for wi in range(0, w_num):
        for hi in range(0, h_num):
            box = (wi * w, hi * h, (wi + 1) * w, (hi + 1) * h)
            piece = im.crop(box)
            tmp_img = Image.new('L', (w, h), 255)
            tmp_img.paste(piece)
            img_path = os.path.join(destination, "%d_%d.png" % (wi, hi))
            tmp_img.save(img_path)
    return os.listdir(destination)
