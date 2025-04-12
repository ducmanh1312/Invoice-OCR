import warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pandas as pd
import sys
sys.path.append(".")
from .CRAFT import CRAFTModel, draw_boxes, draw_polygons, boxes_area, polygons_area

class TextDetection(): 
    def __init__(self, device = 'cuda'):
        self.device = device
        self.model = CRAFTModel(self.device, use_refiner = True, fp16 = True)
    def predict_boxes(self, img):   # predict bounding box
        return self.model.get_boxes(img)    

def draw_boxes(boxes, img, save_path):   # draw bounding box
    draw = ImageDraw.Draw(img)
    for box in boxes:                                                        #(x2,y2)
        x1, y1, x2, y2 = box[0][0], box[0][1],box[2][0], box[2][1]  #(x1,y1)
        if x1 >= x2 or y1 >= y2:
            print(f'Exception: {x1, y1, x2, y2}')
            continue
        draw.rectangle((x1, y1, x2, y2), outline='red')
    if save_path:
        img.save(save_path)
    return img

# Test
if __name__ == '__main__':
    img = Image.open("/media/icnlab/Data/Manh/OCR/test.jpg")
    
    text_det = TextDetection(device = 'cuda')
    boxes = text_det.predict_boxes(img)
    draw_boxes(boxes, img, save_path = None)
    

