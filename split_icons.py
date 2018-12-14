import os
import sys
import glob
from PIL import Image


def use(src_img, dst_path, width, height):
    if not os.path.exists(dst_path) or not os.path.isfile(src_img):
        print 'Not exists', dst_path, src_img
        sys.exit(1)

    w, h = int(width), int(height)
    im = Image.open(src_img)
    im_w, im_h = im.size
    print 'Image width:%d height:%d  will split into (%d %d) ' % (im_w, im_h, w, h)
    w_num, h_num = int(im_w / w), int(im_h / h)

    for wi in range(0, w_num):
        for hi in range(0, h_num):
            box = (wi * w, hi * h, (wi + 1) * w, (hi + 1) * h)
            piece = im.crop(box)
            tmp_img = Image.new('L', (w, h), 255)
            tmp_img.paste(piece)
            img_path = os.path.join(dst_path, "%d_%d.png" % (wi, hi))
            tmp_img.save(img_path)
