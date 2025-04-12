from module.img_segmentation.img_seg import DetectronSegmentation
from module.text_detect.text_det import TextDetection
from module.text_recog.text_recognition import VietOCRPrediction
from module.img_cls.image_classification import EfficientNetClassification
from module.text_cls.PhoBert_prediction import PhoBertPrediction
from module.text_cls.svm_cls import SVMClassifier
import streamlit as st
from PIL import Image, ImageDraw
import cv2
import numpy as np
import pandas as pd
from utils import *

st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50; /* Màu xanh */
        color: white; /* Màu chữ trắng */
    }
    </style>
    """, unsafe_allow_html=True)



# Tiêu đề của ứng dụng
st.title("🎯Trích xuất thông tin từ hoá đơn")
st.sidebar.header("📤Tùy chọn tải hình ảnh lên")

uploaded_file = st.sidebar.file_uploader("Chọn hình ảnh để tải lên:", type=["jpg", "jpeg", "png"])

# Hiển thị nội dung dựa trên việc tải lên hình ảnh
if uploaded_file is not None:
    # Đọc hình ảnh và hiển thị
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Hình ảnh đã tải lên.", use_container_width=True)

# Tao buton send
send_flag = False
if st.sidebar.button('Send'):
    send_flag = True

# Tạo các tab
tabs = st.tabs(["🔥Phase 1", "💧Phase 2", "⚡Phase 3"])

# Nội dung của Tab 1
with tabs[0]:
    done_task1 = False
    detectron = DetectronSegmentation(
        config_path='/media/icnlab/Data/Manh/OCR/Documentation/Detectron2_Segmentation/config.yml',
        weight_path= '/media/icnlab/Data/Manh/OCR/src/module/img_segmentation/weights/model_final.pth',
        device='cuda'
    )
    st.header("Xoá background ảnh 🔨")
    with st.spinner("⏳ Đang xử lý..."):
        if send_flag:
            img = np.array(image)
            img = remove_background(detectron, img)
            st.image(img, caption="Đã xoá background ảnh.", use_container_width=True,  width=50)
            st.success("Xử lý hoàn tất!")
            done_task1 = True
            
# Nội dung của Tab 2
with tabs[1]:
    st.header("Xoay ảnh và phát hiện các box 🎨")
    text_detect_model = TextDetection(device='cuda')
    flip_model = EfficientNetClassification(
        # weight_path = 'weights/weightinvoiced_weight.pth',
    )
    with st.spinner("⏳ Đang xử lý..."):
        done_task2 = False
        if done_task1:
            # Replace
            boxes = text_detect_model.predict_boxes(img)
            # img = rotate_and_flip(flip_model, img, boxes)
            # boxes = text_detect_model.predict_boxes(img)

            draw_boxes(boxes, img, save_path = None)
            st.image(img, caption="Xoay ảnh và phát hiện các box.", use_container_width=True,  width=50)
            st.success("Xử lý hoàn tất!")
            done_task2 = True

# Nội dung của Tab 3
with tabs[2]:
    st.header("Phân loại và hiển thị 🚩")
    
    with st.spinner("⏳ Đang xử lý..."):
        ocr_model  = VietOCRPrediction(
            config_name= 'vgg_transformer',
            weight_path= '/media/icnlab/Data/Manh/OCR/Documentation/Text_Recognize/weights/transformerocr.pth', 
        )
        # Load model PhoBert
        bert_model = PhoBertPrediction(weight_path= '/media/icnlab/Data/Manh/OCR/Documentation/Text_Classification/weights/model_best_valoss.pt')

        # Load model SVM Classification
        svm_clf = SVMClassifier(
            tfidf_path='/media/icnlab/Data/Manh/OCR/Documentation/Text_Classification/svm_model/model_vectorizer_tfidf.pkl',
            svm_path= '/media/icnlab/Data/Manh/OCR/Documentation/Text_Classification/svm_model/svm_classifier_model.pkl'
        )
        if done_task2:
            list_text = text_recog_vietocr(ocr_model, boxes, img) # return list of text
            seller, address, timestamp, totalcost = text_classification(
                phobert_model=bert_model,
                svm_model=svm_clf,
                list_text = list_text,
                boxes=boxes
            )

            text_seller, text_address, text_timestamp, text_totalcost = '', '', '', ''
            for seller_box in seller:
                text_seller += seller_box['text'] +  ' '
            for address_box in address:
                text_address += address_box['text'] + ' '
            for timestamp_box in timestamp:
                text_timestamp += timestamp_box['text'] + ' '
            for totalcost_box in totalcost:
                text_totalcost += totalcost_box['text'] + ' '
            data = {
                'SELLER': [text_seller], 
                'ADDRESS': [text_address],
                'TIMESTAMP': [text_timestamp],
                'TOTALCOST': [text_totalcost]
            }
            df = pd.DataFrame(data)
            st.table(df)
            img = visualize_image1(seller, address, timestamp, totalcost, img)
            st.image(img, caption="Phân loại và hiển thị.", use_container_width=True,  width=50)
            st.success("Xử lý hoàn tất!")
 