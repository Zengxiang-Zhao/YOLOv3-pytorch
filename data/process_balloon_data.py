import json
import copy
import os
import numpy as np
import argparse
import cv2

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = copy.deepcopy(x)
    y[0] = (x[0] + x[2]) // 2
    y[1] = (x[ 1] + x[ 3]) // 2
    y[2] = x[ 2] - x[ 0]
    y[ 3] = x[ 3] - x[ 1]
    return y

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]


            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def write_labels(output_path, dataset_dicts):
    for data in dataset_dicts:

        label_name = os.path.basename(data['file_name']).replace('.jpg','.txt')
        output_file_path = os.path.join(output_path, label_name)
        
        objects = ''
        for box in data['annotations']:
            bbox = xyxy2xywh(box['bbox'])
            str_box = [str(e) for e in bbox]
            objects += str(box['category_id']) +' ' + ' '.join(str_box) + '\n'
        # write to file
        objects = objects.rstrip('\n')
        with open(output_file_path, 'w') as f:
            f.write(objects)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=str, help='output labels path')
	parser.add_argument('--img_dir', type=str, help='balloon data train or valid')
	opt = parser.parse_args()

	print(opt, end='\n\n')

	dataset_dicts = get_balloon_dicts(opt.img_dir)
	write_labels(opt.output, dataset_dicts)

	print('Done')

