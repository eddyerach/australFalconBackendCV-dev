import numpy as np
import cv2
import os
import pandas as pd
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import random


random.seed(5)
output =      'output/'
pre_dilate =  'pre_dilate/'
post_dilate = 'post_dilate/'
bayas =       'bayas/'
elipse =      'elipse/'
solapado =    'solapado/'
no_solapado = 'no_solapado/' 
dos_conteos = 'dos_conteos/'
mascara = 'mascara/'

man1    = {1: 127 ,2: 69 ,3: 144 ,4: 129 ,5: 94 ,6: 78 ,7: 92 ,8: 68 ,9: 88 ,10: 79 ,11: 86 ,12: 72 ,13: 115 ,14: 97 ,15: 103 ,16: 75 ,17: 77 ,18: 94 ,19: 87 ,20: 107 ,21: 148 ,22: 104 ,23: 131 ,24: 97 ,25: 89 ,26: 153 ,27: 119 ,28: 82 ,29: 1405 ,30: 128 ,31: 98 ,32: 122 ,33: 157 ,34: 141 ,35: 144 ,36: 186 ,37: 140 ,38: 86 ,39: 79 ,40: 149 ,41: 259 ,42: 224 ,43: 308 ,44: 239 ,45: 149 ,46: 198 ,47: 223 ,48: 167 ,49: 117 ,50: 238 ,51: 119 ,52: 142 ,53: 144 ,54: 199 ,55: 112 ,56: 141 ,57: 85 ,58: 178 ,59: 109 ,60: 84 ,61: 70 ,62: 118 ,63: 82 ,64: 127 ,65: 128 ,66: 167 ,67: 119 ,68: 103 ,69: 88 ,70: 111 ,71: 104 ,72: 98 ,73: 103 ,74: 96 ,75: 130 ,76: 129 ,77: 117 ,78: 146 ,79: 112 ,80: 109 ,81: 76 ,82: 122 ,83: 108 ,84: 137 ,85: 125 ,86: 131 ,87: 83 ,88: 111 ,89: 121 ,90: 79 ,91: 98 ,92: 106 ,93: 72 ,94: 63 ,95: 99 ,96: 81 ,97: 76 ,98: 71 ,99: 84 ,100: 98}
man2    = {1: 118, 2: 72, 3: 146, 4: 114, 5: 94, 6: 66, 7: 111, 8: 98, 9: 89, 10: 103, 11: 127, 12: 91, 13: 124, 14: 97, 15: 116, 16: 77, 17: 81, 18: 119, 19: 108, 20: 133, 21: 167, 22: 124, 23: 154, 24: 116, 25: 94, 26: 91, 27: 73, 28: 61, 29: 84, 30: 104, 31: 97, 32: 81, 33: 108, 34: 98, 35: 113, 36: 98, 37: 64, 38: 61, 39: 58, 40: 97, 41: 163, 42: 144, 43: 211, 44: 138, 45: 67, 46: 110, 47: 98, 48: 86, 49: 70, 50: 122, 51: 83, 52: 97, 53: 103, 54: 112, 55: 73, 56: 89, 57: 57, 58: 114, 59: 88, 60: 64, 61: 59, 62: 77, 63: 52, 64: 98, 65: 92, 66: 94, 67: 88, 68: 57, 69: 66, 70: 85, 71: 71, 72: 67, 73: 62, 74: 72, 75: 87, 76: 116, 77: 134, 78: 163, 79: 121, 80: 114, 81: 83, 82: 102, 83: 94, 84: 81, 85: 107, 86: 95, 87: 88, 88: 87, 89: 69, 90: 72, 91: 81, 92: 95, 93: 73, 94: 57, 95: 59, 96: 68, 97: 84, 98: 66, 99: 70, 100: 87}
horacio = {1: 119, 2: 138, 3: 157, 4: 104, 5: 139, 6: 110, 7: 118, 8: 127, 9: 90, 10: 131, 11: 170, 12: 88, 13: 246, 14: 181, 15: 193, 16: 113, 17: 118, 18: 169, 19: 189, 20: 71, 21: 78, 22: 107, 23: 120, 24: 159, 25: 119, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 130, 52: 213, 53: 211, 54: 195, 55: 183, 56: 292, 57: 216, 58: 146, 59: 98, 60: 124, 61: 152, 62: 295, 63: 141, 64: 156, 65: 198, 66: 189, 67	: 173, 68: 216, 69: 194, 70: 153, 71: 0, 72: 122, 73: 226, 74: 135, 75: 124, 76: 138, 77: 164, 78: 83, 79: 130, 80: 127, 81: 154, 82: 138, 83: 246, 84: 98, 85: 208, 86: 96, 87: 118, 88: 205, 89: 116, 90: 107, 91: 102, 92: 161, 93: 197, 94: 252, 95: 144, 96: 183, 97: 194, 98: 152, 99: 97, 100: 149}
class BunchDetector:
    def __init__(self, model_path, input_path):
        self.model_path = model_path
        self.input_path = input_path
        self.predictor = None

    def load_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.NUM_GPUS = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.DATASETS.TEST = ("dataset_test",)
        cfg.TEST.DETECTIONS_PER_IMAGE = 200
        self.predictor = DefaultPredictor(cfg)

    def detect_bunches(self):
        output_masks = []
        print('detect_bunches')
        for image in os.listdir(self.input_path):
            filtered_masks = {}
            print(f'Procesando imagen: ', image)
            im = cv2.imread(os.path.join(self.input_path, image))
            outputs = self.predictor(im)
            mask_color = (0, 0, 255)  # Green color for the mask
            masks = outputs["instances"].pred_masks.to("cpu").numpy()
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

            output_image = im.copy()
            cfg = get_cfg()
            cfg.DATASETS.TRAIN = ("my_dataset",)
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            # Get the instances from the output
            instances = outputs["instances"]
            # Draw the masks on the visualizer
            v = v.draw_instance_predictions(instances.to("cpu"))
            # Get the result image
            result_image = v.get_image()[:, :, ::-1]
            cv2.imwrite('result_image.jpg', result_image)
            for mask, box in zip(masks, boxes):
                mask = mask.astype(np.uint8) * 255
                mask_org = mask 
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
                mask = cv2.dilate(mask, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #metodo alternativo: cv2.CHAIN_APPROX_SIMPLE
                #print(f'contornos detectados: {len(contours)}')
                contour_areas = [cv2.contourArea(contour) for contour in contours]
                #print(f'areas detectadas: {contour_areas}')
                #three_channel_mask = cv2.merge([mask] * 3)
                #dibujar el contorno mas grande detectado
                
                #cv2.drawContours(im,contours[contour_areas.index(max(contour_areas))], -1, (0, 0, 0) , 4)
                #cv2.imwrite('three_channel_mask.jpg', im)

                #print(f'contour_areas: {contour_areas}')
                if contour_areas:
                    biggest_mask = contour_areas.index(max(contour_areas))
                    #print(f'biggest_mask id: {biggest_mask}')
                    total_area = contour_areas[biggest_mask]
                    #print(f'biggest_mask area: {total_area}')
                    filtered_masks[total_area] = [contours[biggest_mask], mask]
                else: 
                    print(f'No contour_areas: {image}')
                #print("Total Area:", total_area)
            
            if filtered_masks.keys():
                #print(f'Todas las areas de mascaras: {filtered_masks.keys()}')
                biggest_mask = max(filtered_masks.keys())
                #print(f'Mascara mas grande: {biggest_mask}')
                final_contours = biggest_mask
                biggest_contour = filtered_masks[biggest_mask][0]
                #print(f'biggest_contour: {biggest_contour}')
                mask = filtered_masks[biggest_mask][1]
                #dilatation_dst = cv.dilate(src, kernel)
                new_mask = np.expand_dims(mask, axis=-1)
                new_mask = np.repeat(new_mask, 3, axis=-1)
                mask_org_exp = np.expand_dims(mask_org, axis=-1)
                mask_org_exp = np.repeat(mask_org_exp, 3, axis = -1)
                masked_detection_org_mask = cv2.bitwise_and(im,mask_org_exp, mask=mask_org) ##Extracto de la mascara en imagen rgb
                cv2.imwrite(output+pre_dilate+image, masked_detection_org_mask)

                #new_mask = np.array([mask.astype(np.uint8) * 255,mask.astype(np.uint8) * 255,mask.astype(np.uint8) * 255])
                #print(f'im type : {type(im)} {im.shape}, new_mask type: {type(new_mask)} {new_mask.shape}')
                masked_detection = cv2.bitwise_and(im,new_mask, mask=mask) ##Extracto de la mascara en imagen rgb
                cv2.imwrite(output+post_dilate+image, masked_detection)
                black_image = np.zeros_like(im)
                black_image[filtered_masks[biggest_mask][1] != 0] = masked_detection[filtered_masks[biggest_mask][1] != 0]
                output_masks.append([image,black_image,final_contours,biggest_mask,biggest_contour])
                output_mask_path = 'output_masks/'

        return output_masks


class GrapesDetector:
    def __init__(self, model_path, input_masks, input_images):
        self.model_path = model_path
        self.input_masks = input_masks
        self.input_images = input_images
        self.predictor = None

    def load_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.DATASETS.TEST = ("dataset_test",)
        cfg.TEST.DETECTIONS_PER_IMAGE = 200
        self.predictor = DefaultPredictor(cfg)

    def detect_grapes(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        df = pd.DataFrame()
        filtered_masks = []
        image_grapes_masks = []
        
        print(f'En detect_grapes. len(self.input_masks): {len(self.input_masks)}')
        for idx, image in enumerate(self.input_masks):
            mask_output_path = output + bayas + mascara + image[0]
            print(f'Writting mask image: {mask_output_path}')
            cv2.imwrite(mask_output_path, image[1])
            #print(f'Loading org image: {os.path.join(self.input_images, image_org)}')
            org_image = cv2.imread(os.path.join(self.input_images, image[0]))
            outputs = self.predictor(image[1])
            contour_racimo = image[2]
            mask_color = (0, 0, 255)  # Green color for the mask
            masks = outputs["instances"].pred_masks.to("cpu").numpy()
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            #output_image = im.copy()
            grapes_masks = []
            area_bayas_imagen = 0
            area_total_no_solapada = np.zeros_like(org_image)[:,:,0]
            #print(f'mascara racimo: {contour_racimo}')
            cv2.drawContours(org_image,image[4], -1, (0, 0, 0) , 4) ##Dibujar racimo con borde negro
            im_racimo_bayas_no_overlap = org_image.copy()
            print(f'*** bayas detectadas: {len(boxes)}')
            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                area_total_no_solapada += mask
                area_total_no_solapada = np.clip(area_total_no_solapada, 0, 1)
                mask = mask.astype(np.uint8) * 255
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour_areas = [cv2.contourArea(contour) for contour in contours]
                biggest_mask = contour_areas.index(max(contour_areas))
                grapes_masks.append(mask)
                total_area = contour_areas[biggest_mask]
                area_bayas_imagen += total_area
                #cv2.drawContours(im_racimo_bayas_no_overlap, contours[biggest_mask], -1, mask_color, 4)#area solapada en rojo
                #print(f'box: {boxes[idx]}')
                im_racimo_bayas_no_overlap = cv2.putText(im_racimo_bayas_no_overlap, str(idx), (int(boxes[idx][0]),int(boxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2, cv2.LINE_AA)
                cv2.drawContours(im_racimo_bayas_no_overlap, contours, -1, mask_color, 4)#area solapada en rojo
                score_height = (int(box[0]), int(box[1]) - 5) 
            #print(f'dibujando {len()}')
            num_grapes = 0
            num_grapes = len(masks)
            font_scale = 1
            thickness = 2
            contours1, hierarchy1 = cv2.findContours(area_total_no_solapada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #[cv2.contourArea(contour) for contour in contours]
            im_racimo_bayas_no_overlap = cv2.rectangle(im_racimo_bayas_no_overlap, (0,0), (im_racimo_bayas_no_overlap.shape[1],int(im_racimo_bayas_no_overlap.shape[0]/3.3)), (0,0,0), -1)
            not_solaped_contour = sum([cv2.contourArea(contour1) for  contour1 in contours1])
            im_racimo_bayas_no_overlap = cv2.putText(im_racimo_bayas_no_overlap, 'Num bayas: '      +str(num_grapes), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            im_racimo_bayas_no_overlap = cv2.putText(im_racimo_bayas_no_overlap, 'Area racimo: '    +str('{:.2f}'.format(image[3]/1000)) + 'K', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            im_racimo_bayas_no_overlap = cv2.putText(im_racimo_bayas_no_overlap, 'Area bayas s.: '  +str('{:.2f}'.format(area_bayas_imagen/1000)) + 'K', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            im_racimo_bayas_no_overlap = cv2.putText(im_racimo_bayas_no_overlap, 'Area bayas N.s.: '+str('{:.2f}'.format(not_solaped_contour/1000))+'K', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            if (image[3] == 0 or area_bayas_imagen == 0):
                ratio_ba = 'NAN'
            else:
                ratio_ba = '{:.2f}'.format(area_bayas_imagen/image[3])
            
            if (image[3] == 0 or not_solaped_contour == 0):
                ratio_ba_ns = 'NAN'
            else:
                ratio_ba_ns = '{:.2f}'.format(not_solaped_contour/image[3])
            im_racimo_bayas_no_overlap = cv2.putText(im_racimo_bayas_no_overlap, 'idx. comp. solap.: '+str(ratio_ba), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            im_racimo_bayas_no_overlap = cv2.putText(im_racimo_bayas_no_overlap, 'idx. comp. N. solap.: '+str(ratio_ba_ns), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            cv2.imwrite('output/' + bayas + solapado + image[0], im_racimo_bayas_no_overlap.astype(np.float32))
            print(f'Writting final image: {image[0]}')
            contours, hierarchy = cv2.findContours(area_total_no_solapada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print(f'**contours: {len(contours)}')
            cv2.drawContours(org_image,contours, -1, (255, 0, 0), 4)
            output1 = 'output/' + bayas + no_solapado+ image[0]
            #cv2.imwrite(output1, org_image.astype(np.float32)) ##se guarda imagen de bayas no solapadas
            split_num = int(random.random() * 10)
            if grapes_masks:
                cont_horacio = int(horacio[int(image[0].split('.')[0][:-1])])
                cont_man1 = int(man1[int(image[0].split('.')[0][:-1])])
                filtered_masks.append(
                    {'imagen': image[0], 'area_racimo':image[3], 'area_bayas': area_bayas_imagen, 
                     'area_bayas_ns':not_solaped_contour, 'idx_sol': area_bayas_imagen/image[3],
                     'idx_no_sol':not_solaped_contour/image[3], 'det_bayas': len(grapes_masks), 
                     'man1': int(man1[int(image[0].split('.')[0][:-1])]),'man2':int(man2[int(image[0].split('.')[0][:-1])]),
                     'horacio':int(horacio[int(image[0].split('.')[0][:-1])]),
                     'horacio_man1': cont_horacio if cont_horacio > cont_man1 else cont_man1,  
                     'split': 'test' if split_num < 2 else 'train'})
            else:
                print(f'not grapes masks')
                filtered_masks.append({'imagen': image[0], 'area_racimo':image[3], 'area_bayas': area_bayas_imagen, 'area_bayas_ns':not_solaped_contour, 'det_bayas': len(grapes_masks)})

        df = pd.DataFrame.from_records(filtered_masks)
        df.to_csv('output/area_racimos-bayas_0614_sep10_th01.csv')


def main():
    input_path = 'input/dataset/0614_b'
    #input_path = 'input/dataset/tm'
    #input_path = 'input/dataset/test1'
    #input_path = 'input/dataset/test'
    #input_path = 'input/dataset/0627'
    output_path = 'output/'
    model_grapes = 'modelos/model_v3.pth'
    #model_bunches = 'model_final_evidentes_aumentado_dropout_color_rotacion_90kit.pth'
    model_bunches = 'model_final.pth'

    bunch_detector = BunchDetector(model_bunches, input_path)
    bunch_detector.load_model()
    bunch_masks = bunch_detector.detect_bunches()
    #print(f'bunch_masks: {bunch_masks}')
    
    grapes_detector = GrapesDetector(model_grapes, bunch_masks, input_path)
    grapes_detector.load_model()
    grapes_masks = grapes_detector.detect_grapes()


if __name__ == "__main__":
    main()