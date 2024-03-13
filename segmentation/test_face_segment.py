import cv2, json
import numpy as np
from pathlib import Path

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.projects import point_rend


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_mask(outputs, class_dict):

    mask_image = np.zeros([outputs['instances'].image_size[0], outputs['instances'].image_size[1], 3], np.uint8)

    draw_list = []
    class_array = outputs['instances'].pred_classes.cpu().numpy()
    print(class_array)

    # draw 순서 정하기
    for class_id in class_dict:
        print(class_id)
        index_list = np.where(class_array == int(class_id))[0]
        draw_list += index_list.tolist()

    for i in draw_list:
        class_id = class_array.tolist()[i]
        mask_image[(outputs['instances'].pred_masks[i]).cpu().numpy()] = class_dict[str(class_id)][1]


    return mask_image


if __name__ == '__main__':


    register_coco_instances(
        'my_dataset',
        {},
        './segmentation/dataset/_annotations.coco.json',
        './segmentation/dataset',
    )

    with open(
        './segmentation/dataset/_annotations.coco.json',
        'r',
    ) as json_file:
        json_data = json.load(json_file)
        num_of_class = len(json_data['categories'])
        categories = json_data['categories']
        print('num_of_class: ', num_of_class)



    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.MODEL.WEIGHTS = './segmentation/model/Retina_model.pth'  # model
    cfg.MODEL.DEVICE = 'cuda:0'

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Exposure threshold based on accuracy
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = num_of_class
    predictor = DefaultPredictor(cfg)  # Basic prediction
    test_metadata = MetadataCatalog.get('my_dataset')
    dataset_dicts = DatasetCatalog.get('my_dataset')  # Metadata registration and catalog registration (for class classification)


    # In the order of drawing the RGB model class dictionary 
    class_dict_RGB = {
        '10': ['face', hex_to_rgb('#44690C')],
        '6': ['nose', hex_to_rgb('#569A93')],
        '7': ['upper_lip', hex_to_rgb('#A6480A')],
        '8': ['lower_lip', hex_to_rgb('#F25F41')],
        '0': ['eye_brow', hex_to_rgb('#F19BDC')],
        '4': ['LWA', hex_to_rgb('#0000FF')],
        '5': ['M-C', hex_to_rgb('#DE9846')],
        '3': ['caruncle', hex_to_rgb('#F4F812')],
        '2': ['iris', hex_to_rgb('#805472')],
        '1': ['double_eyelid', hex_to_rgb('#D60ACF')],
        '9': ['maker', hex_to_rgb('#7C3E3D')],
        '11': ['double_eyelid_2', hex_to_rgb('#ff7f00')],
        '12': ['double_eyelid_3', hex_to_rgb('#ffff00')],
        '13': ['double_eyelid_4', hex_to_rgb('#008000')],
        '14': ['double_eyelid_5', hex_to_rgb('#0067a3')]
    }


    im_input_path = './segmentation/input'
    im_list = ['0.png', '1.png']

    im_output_path = './segmentation/output'
    Path(im_output_path).mkdir(exist_ok=True, parents=True)

    for im_name in im_list:
        imageName = im_input_path + '/' + im_name
        im = cv2.imread(imageName)
        img_h, img_w, _ = im.shape
        print(im_name)

        outputs = predictor(im)  # generate predictor


        # generate overlay image
        # v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=1.0)
        # out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        # cv2.imwrite(f'{im_output_path}/{im_name}', out.get_image()[:, :, ::-1])

        # generate mask image 
        im_mask = get_mask(outputs, class_dict_RGB)
        cv2.imwrite(f'{im_output_path}/{im_name}', im_mask)



