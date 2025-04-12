from module.img_segmentation.img_seg import DetectronSegmentation
from module.text_detect.text_det import TextDetection
from module.text_recog.text_recognition import VietOCRPrediction
from module.img_cls.image_classification import EfficientNetClassification
from module.text_cls.PhoBert_prediction import PhoBertPrediction
from module.text_cls.svm_cls import SVMClassifier
from utils import remove_background, detection_and_rotate, text_classification, text_recog_vietocr, visualize_image
import cv2

print('====================================LOAD MODEL================================')
# Load model Detectron2
detectron_cfg_path = './module/img_segmentation/config.yml'
segment = DetectronSegmentation(config_path=detectron_cfg_path)

# Load model Image Classificationn
img_clf_weight = 'weights/weightinvoiced_weight.pth'
efficientnet_model = EfficientNetClassification()
efficientnet_model.load_weight(img_clf_weight)

# Load model CRAFT for Text Detection
text_det = TextDetection(device='cuda')

# Load model VietOCR
ocr_model  = VietOCRPrediction(
        config_name= 'vgg_transformer',
        weight_path= './weights/transformerocr.pth', 
    )

# Load model PhoBert
bert_model = PhoBertPrediction(3)
bert_model.load_weight( weight_path= 'weights/model_best_valoss.pt')

# Load model SVM Classification
svm_clf = SVMClassifier(
    tfidf_path='weights/model_vectorizer_tfidf.pkl',
    svm_path= 'weights/svm_classifier_model.pkl'
)


print('=========================DONE==========================================')

if __name__ == '__main__':
    img_path  = '../dataset/val_images/mcocr_val_145114aszbc.jpg'
    img = cv2.imread(img_path)
    img = remove_background(segment, img)
    img, boxes = detection_and_rotate(text_det, efficientnet_model, img)
    text = text_recog_vietocr(ocr_model, boxes, img)
    seller, address, timestamp, totalcost = text_classification(
        phobert_model=bert_model,
        svm_model=svm_clf,
        list_text = text,
        boxes=boxes
    )
    print(seller, address, timestamp, totalcost)
    visualize_image('anh.jpg', seller, address, timestamp, totalcost, img)


        