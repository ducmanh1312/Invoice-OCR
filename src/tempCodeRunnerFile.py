boxes = text_detect_model.predict_boxes(img)
img = rotate_and_flip(flip_model, img, boxes)
boxes = text_detect_model.predict_boxes(img)