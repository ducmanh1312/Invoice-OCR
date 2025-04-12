from module.img_segmentation.img_seg import DetectronSegmentation
from module.text_detect.text_det import TextDetection
from module.text_recog.text_recognition import VietOCRPrediction
from module.img_cls.image_classification import EfficientNetClassification
from module.text_cls.PhoBert_prediction import PhoBertPrediction
from module.text_cls.svm_cls import SVMClassifier
from utils import remove_background, detection_and_rotate, text_recog_vietocr, text_classification, PipeLineProcessImage




print('======================================================LOAD MODEL================================================================')
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
print('======================================================================================================================')


pipeline = PipeLineProcessImage(
    segment_model=segment,
    detect_model=text_det,
    imgclf_model= efficientnet_model,
    vietocr_model= ocr_model,
    phobert_model=bert_model,
    svm_model=svm_clf
)

def img_process(img):
    seller, address, timestamp, totalcost = pipeline.pipe(img)
    seller_text, address_text, timestamp_text, totalcost_text = '', '', '', ''
    for seller_box in seller:
        seller_text += seller_box['text']
    for address_box in address:
        address_text +=  address_box['text']
    for timestamp_box in timestamp:
        timestamp_text += timestamp_box['text']
    for totalcost_box in totalcost:
        totalcost_text += totalcost_box['text']
    return {'seller': seller_text,
            'address': address_text,
            'timestamp': timestamp_text,
            'total_cost': totalcost_text}


