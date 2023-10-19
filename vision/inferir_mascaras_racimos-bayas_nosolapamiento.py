import numpy as np
import cv2
import os
import pandas as pd
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

def calculate_area(binary_mask):
    return np.sum(binary_mask)  # Count the number of white pixels

def add_areas_without_intersection(detection_masks):
    #print(f'In detection masks: {detection_masks}')
    total_area = 0
    non_overlapping_mask = np.zeros_like(detection_masks[0])

    for mask in detection_masks:
        # Subtract the intersection with the previously processed masks
        intersection = np.logical_and(mask, np.logical_not(non_overlapping_mask))
        mask_area = calculate_area(mask) - calculate_area(intersection)
        
        total_area += mask_area
        non_overlapping_mask = np.logical_or(non_overlapping_mask, mask)

    return total_area

class BunchDetector:
    def __init__(self, model_path, input_path):
        self.model_path = model_path
        self.input_path = input_path
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

    def detect_bunches(self):
        output_masks = []

        for image in os.listdir(self.input_path):
            filtered_masks = {}
            print('IMAGE:', image)
            im = cv2.imread(os.path.join(self.input_path, image))
            im2 = cv2.imread(os.path.join(self.input_path, image))
            mask = np.zeros_like(im2)
            outputs = self.predictor(im)
            mask_color = (0, 0, 255)  # Green color for the mask
            masks = outputs["instances"].pred_masks.to("cpu").numpy()
            boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

            output_image = im.copy()

            for mask, box in zip(masks, boxes):
                mask = mask.astype(np.uint8) * 255
                #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                #print(f'polygonos detectados: {len(contours)}')
                contour_areas = [cv2.contourArea(contour) for contour in contours]
                three_channel_mask = cv2.merge([mask] * 3)
                cv2.imwrite('three_channel_mask.jpg', three_channel_mask)

                #print(f'contour_areas: {contour_areas}')
                if contour_areas:
                    biggest_mask = contour_areas.index(max(contour_areas))
                    total_area = contour_areas[biggest_mask]
                    filtered_masks[total_area] = [contours[biggest_mask], mask]
                #print("Total Area:", total_area)
            
            if filtered_masks.keys():
                #print('filtered', filtered_masks.keys())
                biggest_mask = max(filtered_masks.keys())
                #final_contours = filtered_masks[biggest_mask][0]
                final_contours = biggest_mask
                biggest_contour = filtered_masks[biggest_mask][0]
                #print(f'dibujando mascara: {biggest_mask}')
                ##output_masks.append(cv2.bitwise_and(image, final_contours))
                #cv2.drawContours(output_image, final_contours, -1, mask_color, 4)
                #out = np.zeros_like(im2)
                #print(f'out {len(out)},{len(out[0])}')
                #print(f'im2 {len(im2)},{len(im2[0])}')
                #mask = np.zeros_like(im2) #
                #out[final_contours == 255] = im2[final_contours == 255]
                ##dict_area.append({'image': image, 'area': biggest_mask})
                #output = os.path.join('output/', image)
                #mask = np.zeros_like(im2)
                #cv2.drawContours(mask, final_contours, -1, mask_color, 4)
                #out = np.zeros_like(im2) # Extract out the object and place into output image
                #out[mask == 255] = im2[mask == 255]
                ## Now crop
                #(y, x) = np.where(mask == 255)
                #(topy, topx) = (np.min(y), np.min(x))
                #(bottomy, bottomx) = (np.max(y), np.max(x))
                #out = out[topy:bottomy+1, topx:bottomx+1]
                #cv2.imwrite(output, output_image.astype(np.float32))
                #result = im2.copy()
                #result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
                #result[:, :, 3] = np.zeros_like(im2)
                #result[:, :, 3] = mask
                #mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ###incrementar tamano mascara
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
                ###filtered_masks[biggest_mask][1] = cv2.dilate(filtered_masks[biggest_mask][1], kernel, iterations=1)
                ###masked_detection = cv2.bitwise_and(im, im, mask=filtered_masks[biggest_mask][1])

                #masked_detection = cv2.bitwise_and(im,im, mask=mask)
                ##nonew_mask = expanded_array = np.repeat(mask.astype(np.uint8) * 255[:, :, np.newaxis], 3, axis=2)
                #no #new_mask = np.stack([mask.astype(np.uint8) * 255] * 3, axis=0)
                # Expand the dimensions to create a 3-channel mask
                new_mask = np.expand_dims(mask, axis=-1)
                new_mask = np.repeat(new_mask, 3, axis=-1)
                #new_mask = np.array([mask.astype(np.uint8) * 255,mask.astype(np.uint8) * 255,mask.astype(np.uint8) * 255])
                print(f'im type : {type(im)} {im.shape}, new_mask type: {type(new_mask)} {new_mask.shape}')
                masked_detection = cv2.bitwise_and(im,new_mask, mask=mask)
                #cv2.imwrite('test.jpg', new_mask)
                black_image = np.zeros_like(im)
                black_image[filtered_masks[biggest_mask][1] != 0] = masked_detection[filtered_masks[biggest_mask][1] != 0]
                #print(f'black_image : {type(black_image)}, {black_image}')
                #cv2.imwrite('cutout_image.jpg', black_image)

                output_masks.append([image,black_image,final_contours,biggest_mask,biggest_contour])
                # save resulting masked image
                output_mask_path = 'output_masks/'
                #cv2.imwrite(output_mask_path + image, black_image)
            #else:
            #    output_masks.append([image,0,0,0])


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
            print(f'Writting mask image: {image[0]}')
            mask_output_path = 'output_mask/' + image[0]
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
            cv2.drawContours(org_image,image[4], -1, (0, 0, 0) , 4)
            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                #print(f'tamanos {area_total_no_solapada.shape}, {mask.shape}')
                area_total_no_solapada += mask
                area_total_no_solapada = np.clip(area_total_no_solapada, 0, 1)
                #print(f'mask: {mask}')
                mask = mask.astype(np.uint8) * 255
                #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                #print(f'polygonos detectados: {len(contours)} hierarchy: {hierarchy}')
                contour_areas = [cv2.contourArea(contour) for contour in contours]
                biggest_mask = contour_areas.index(max(contour_areas))
                grapes_masks.append(mask)
                #print(f'contours: {contours}')
                #cv2.drawContours(output_image, contours[biggest_mask], -1, mask_color, 4)
                #cv2.drawContours(output_image, contours, -1, mask_color, 4)
                #biggest_mask = contour_areas.index(max(contour_areas))
                #cv2.rectangle(output_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), mask_color, 2)
                #total_area = sum(contour_areas)
                total_area = contour_areas[biggest_mask]
                area_bayas_imagen += total_area
                #print(f'total_area: {total_area}')
                cv2.drawContours(org_image, contours[biggest_mask], -1, mask_color, 4)
                ####cv2.drawContours(org_image, contour_racimo, -1, mask_color, 4)
                score_height = (int(box[0]), int(box[1]) - 5) 
                #print(f'score_height: {score_height}')
                #im = cv2.putText(im,str(10) + str(idx),score_height, font, 0.3,(0,0,255),1,cv2.LINE_AA)
                #output_image = cv2.putText(output_image,str(total_area) + str(idx),score_height, font, 1,(0,0,255),1,cv2.LINE_AA)
                #total_area = contour_areas[biggest_mask]
                #print("Contour Areas:", contour_areas)
                #filtered_masks[total_area] = contours[biggest_mask]
                #filtered_masks[idx] = total_area
                #print("Total Area:", total_area)
                
                #print(f'In for len masks: {len(grapes_masks)}')
            #print(f'Sumar area de {len(grapes_masks)}')

            #dibujando mascara final: 
            print(f'Writting final image: {image[0]}')
            contours, hierarchy = cv2.findContours(area_total_no_solapada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #print(f'contours: {contours}')
            cv2.drawContours(org_image,contours, -1, (255, 0, 0), 4)
            output = os.path.join('output/', image[0])
            cv2.imwrite(output, org_image.astype(np.float32))
            if grapes_masks:
                #detection_masks = grapes_masks  # Replace mask1, mask2, mask3 with your detected masks
                #total_area_without_intersection = add_areas_without_intersection(detection_masks)
                #print("Total area without intersection:", total_area_without_intersection)
                filtered_masks.append({'imagen': image[0], 'area_racimo':image[3], 'area_bayas': area_bayas_imagen, 'det_bayas': len(grapes_masks)})
            else:
                print(f'not grapes masks')
                filtered_masks.append({'imagen': image[0], 'area_racimo':image[3], 'area_bayas': area_bayas_imagen, 'det_bayas': len(grapes_masks)})

        df = pd.DataFrame.from_records(filtered_masks)
        df.to_csv('area_racimos-bayas_0627_hoy.csv')


def main():
    #input_path = 'input/dataset/0614_b'
    input_path = 'input/dataset/test'
    #input_path = 'input/dataset/0627_b'
    output_path = 'output/'
    model_grapes = 'modelos/model_v3.pth'
    #model_bunches = 'model_final_evidentes_aumentado_dropout_color_rotacion_90kit.pth'
    model_bunches = 'modelos/model_final.pth'

    bunch_detector = BunchDetector(model_bunches, input_path)
    bunch_detector.load_model()
    bunch_masks = bunch_detector.detect_bunches()
    #print(f'bunch_masks: {bunch_masks}')
    
    grapes_detector = GrapesDetector(model_grapes, bunch_masks, input_path)
    grapes_detector.load_model()
    grapes_masks = grapes_detector.detect_grapes()



if __name__ == "__main__":
    main()
