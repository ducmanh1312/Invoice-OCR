import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np
from module.text_cls.regex import regex
import re
from PIL import Image, ImageDraw, ImageFont

###
from module.img_segmentation.img_seg import DetectronSegmentation
from module.text_detect.text_det import TextDetection
from module.text_recog.text_recognition import VietOCRPrediction
from module.img_cls.image_classification import EfficientNetClassification
from module.text_cls.PhoBert_prediction import PhoBertPrediction
from module.text_cls.svm_cls import SVMClassifier
###


def remove_background(model, img: list) -> Image:
    img = model.predict(img)
    img = Image.fromarray(img)
    return img

def draw_boxes(boxes, img, save_path=None):   # draw bounding box
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

def rotation_image(boxes, img):
    # Compute angle for rotation
    boxes_ratio = []
    direct_roate = []
    total_width = []
    total_height = []
    for box in boxes:
        x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[1][1]
        total_width.append(x2 - x1)
        total_height.append(y2 - y1)
        box_ratio = abs(y1 - y2)  / abs(x1 - x2)
        direct_roate.append(y1 - y2)
        boxes_ratio.append(box_ratio)
    mean_ratio = np.mean(boxes_ratio)
    mean_direct_rotate = np.mean(direct_roate)
    angle = np.arctan(mean_ratio)
    if mean_direct_rotate >= 0:
        angle_degree = -np.degrees(angle)
    else:
        angle_degree  = np.degrees(angle)
    # Rotate image
    rotated_image = img.rotate(angle_degree, expand = True)
    if np.mean(total_width) < np.mean(total_height):
        rotated_image = img.rotate(90, expand = True)
    # rotated_image.save('abc.jpg')
    return rotated_image

def rotate_and_flip(flip_model, img, boxes):
    rotated_img = rotation_image(boxes, img)
    isfip = flip_model.predict(rotated_img)
    if isfip:
        rotated_img = rotated_img.rotate(180)
    # boxes = text_detect_model.predict_boxes(rotated_img)
    return rotated_img

def detection_and_rotate(detect_model,imgclf_model, img):
    boxes = detect_model.predict_boxes(img)
    rotated_img = rotation_image(boxes, img)
    check = imgclf_model.predict(rotated_img)
    if check:
        rotated_img = rotated_img.rotate(180)
    boxes = detect_model.predict_boxes(rotated_img)
    return rotated_img, boxes

# # test
# from module.text_detect.text_det import TextDetection
# from module.img_cls.image_classification import EfficientNetClassification

# img = Image.open("/media/icnlab/Data/Manh/OCR/Documentation/Detectron2_Segmentation/Invoice_Segmentation-7/valid/mcocr_public_145013alpyf_jpg.rf.7043fab34d177f9da3cc10143dc32dba.jpg")
# text_detect_model = TextDetection(device='cuda')
# flip_model = EfficientNetClassification(
#     device = 'cuda'
# )

# boxes = text_detect_model.predict_boxes(img)
# img = rotate_and_flip(flip_model, img, boxes)
# boxes = text_detect_model.predict_boxes(img)

# # boxes, img = detection_and_rotate(text_detect_model,flip_model,img)
# img = draw_boxes(boxes, img)
# img.show()



def text_recog_vietocr(vietocr_model, boxes, img):
    text = []
    for box in boxes:
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        sub_img = img.crop((x1, y1, x2, y2))
        text.append(vietocr_model.predict(sub_img))
    return text

def text_classification(phobert_model, svm_model, list_text, boxes):
    text_phobert = list_text[:10] 
    text_svm = list_text[10:]
    seller_class = []
    address_class = []
    # PhoBert for clf class address, seller
    for i, text_i in enumerate(text_phobert): 
        res = phobert_model.predict(text_i)
        if res == 0:
            seller_class.append(
                {
                    'text': text_i,
                    'bbox': boxes[i]
                }
            )
        if res == 1:
            address_class.append(
                {
                    'text': text_i,
                    'bbox': boxes[i]
                }
            )
    
    # SVM for clf
    totalcost_class = []    # 
    for i, text_i in enumerate(text_svm):
        res = svm_model.predict(text_i)
        if res == 0:
            totalcost_class.append(
                {
                    'text': text_i,
                    'bbox': boxes[i+10]
                }
            )
            totalcost_class.append(
                {
                    'text': text_svm[i+1],
                    'bbox': boxes[i + 1 + 10]
                }
            )
            break
    timestamp_class = []
    for i, text_i in enumerate(list_text):
        if regex(text_i):
            match = re.search(r'\b' + re.escape('Ngày') + r'\b', text_i, re.IGNORECASE)
            if match:
                text_i = text_i[match.start():]
            timestamp_class.append(
                {
                    'text': text_i,
                    'bbox':  boxes[i]
                }
            )
    return seller_class, address_class, timestamp_class, totalcost_class

def visualize_image(save_path, seller, address, timestamp, totalcost, img):
    draw = ImageDraw.Draw(img)
    for seller_box in seller:
        text = seller_box['text']
        box = seller_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'SELLER', fill="red")
    for address_box in address:
        text = address_box['text']
        box = address_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'ADDRESS', fill="red")
    for timestamp_box in timestamp:
        text = timestamp_box['text']
        box = timestamp_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'TIMESTAMP', fill="red")
    for totalcost_box in totalcost:
        text = totalcost_box['text']
        box = totalcost_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'TOTALCOST', fill="red")
    
    img.save(save_path)

def visualize_image1(seller, address, timestamp, totalcost, img):
    draw = ImageDraw.Draw(img)
    for seller_box in seller:
        text = seller_box['text']
        box = seller_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'SELLER', fill="red")
    for address_box in address:
        text = address_box['text']
        box = address_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'ADDRESS', fill="red")
    for timestamp_box in timestamp:
        text = timestamp_box['text']
        box = timestamp_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'TIMESTAMP', fill="red")
    for totalcost_box in totalcost:
        text = totalcost_box['text']
        box = totalcost_box['bbox']
        x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
        draw.rectangle([(x1, y1), (x2, y2)], outline='blue', width=2)
        draw.text((x1, y1 - 10), 'TOTALCOST', fill="red")
    return img

class PipeLineProcessImage():
    def __init__(self, segment_model, detect_model, imgclf_model, vietocr_model, phobert_model, svm_model):
        self.segmentation_model = segment_model
        self.detect_model = detect_model
        self.imgclf_model = imgclf_model
        self.ocr_model = vietocr_model
        self.phobert_model = phobert_model
        self.svm_model = svm_model
    def pipe(self, img):
        img = remove_background(self.segmentation_model, img)
        img, boxes = detection_and_rotate(self.detect_model, self.imgclf_model, img)
        text = text_recog_vietocr(self.ocr_model, boxes, img)
        seller, address, timestamp, totalcost = text_classification(self.phobert_model, self.svm_model, text, boxes)
        return seller, address, timestamp, totalcost

