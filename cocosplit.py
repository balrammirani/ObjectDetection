import json
import argparse
import funcy
import os
from sklearn.model_selection import train_test_split

def save_coco(file, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def splitdata(filepath,is_annotated,split_ratio):
    with open(filepath, 'rt', encoding='UTF-8') as annotations:
        file_dir = os.path.dirname(filepath)
        coco = json.load(annotations)
        
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if is_annotated:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=split_ratio, random_state=42)

        save_coco(os.path.join(file_dir,"train.json"), licenses, x, filter_annotations(annotations, x), categories)
        save_coco(os.path.join(file_dir,"test.json"),  licenses, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), 'train.json', len(y), 'test.json'))


