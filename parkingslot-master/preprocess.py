import xml.etree.ElementTree as ET
import numpy as np
import os
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.config import Config
import mrcnn.model as modellib


class PreprocessPKLOT(Dataset):

    def load_data(self, dir, is_train=True):
        self.add_class('parking',1, 'Ocupado')
        self.add_class('parking',0, 'Libre')
        if is_train:
            img_dir = os.path.join(dir, 'train/images')
            labels_dir = os.path.join(dir, 'train/labels')
        elif not is_train:
            img_dir = os.path.join(dir, 'test/images')
            labels_dir = os.path.join(dir, 'test/labels')
        for filename in os.listdir(img_dir):
            image_id = filename[:-4]
            img_path = os.path.join(img_dir, filename)
            label_path = labels_dir + '/' + image_id + '.xml'
            self.add_image('parking', image_id=image_id, path=img_path, annotation=label_path)

    def extract_xml_contours(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        contours = []
        for space in root.getiterator('space'):
            type_id = int(space.attrib['occupied'])
            for rect in space.findall('rotatedRect/center'):
                x = int(rect.get('x'))
                y = int(rect.get('y'))
            for rect in space.findall('rotatedRect/size'):
                w = int(rect.get('w'))
                h = int(rect.get('h'))
            for rect in space.findall('rotatedRect/angle'):
                d = int(rect.get('d'))
            coords = [type_id, x,y,w,h,d]
            contours.append(coords)
        return contours

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        contours = self.extract_xml_contours(path)
        mask = np.zeros([720,1280,len(contours)], dtype='uint8')
        class_ids = []
        for i in range(len(contours)):
            cont = contours[i]
            type_id, x,y,w,h,d = cont
            row_s, row_e = int(y-(w)),int(y+(w/2))
            col_s, col_e = int(x-(h/2)), int(x+(h/2))
            mask[row_s:row_e, col_s:col_e,i] = 1
            if type_id == 1:
                class_ids.append(self.class_names.index('Ocupado'))
            else:
                class_ids.append(self.class_names.index('Libre'))
        return mask, np.asarray(class_ids, dtype='int32')
    
    def image_ref(self, image_id):
        info = self.image_info[image_id]
        return info['path']

class ParkingConfig(Config):
    NAME = 'parking_cfg'
    NUM_CLASSES = 3 # porque hay que añadir el background mas las dos clases (ocupado y vacio)
    STEPS_PER_EPOCH = 50

if __name__ == '__main__':
    COCO_MODEL_PATH = os.path.join('.', "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

    train_set = PreprocessPKLOT()
    train_set.load_data('parkingslot-master\ParkingOcuppied', is_train=True)
    train_set.prepare()
    test_set = PreprocessPKLOT()
    test_set.load_data('parkingslot-master\ParkingOcuppied', is_train=False)
    test_set.prepare() 
    ParkingConfig().display()
    model = modellib.MaskRCNN(mode='training', model_dir=COCO_MODEL_PATH, config=ParkingConfig())
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_mask'])
    model.train(train_set, test_set, learning_rate=ParkingConfig().LEARNING_RATE, epochs=5, layers='heads')

