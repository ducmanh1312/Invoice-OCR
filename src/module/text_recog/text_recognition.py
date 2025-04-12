import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer


class VietOCRTrainerwithCustomDataset():
    def __init__(self, config_name):
        self.config = Cfg.load_config_from_name(config_name)
    def set_config(self, dataset_params, params):
        self.config['trainer'].update(params)
        self.config['dataset'].update(dataset_params)
        self.config['device'] = 'cuda:0'
    def train(self, save_path):
        self.trainer = Trainer(self.config, pretrained=True)
        # Save config
        self.trainer.save(save_path)
        self.trainer.train()


class VietOCRPrediction():
    def __init__(self, config_name, weight_path, device = 'cuda:0'):
        self.config = Cfg.load_config_from_name(config_name)
        self.config['weight'] = weight_path   ## đường dẫn đến trọng số đã huấn luyện hoặc comment để sử dụng pretrained model
        self.config['cnn']['pretrained'] = False
        self.config['device'] = device
        self.detector = Predictor(self.config)
    def predict(self, img):
        res = self.detector.predict(img)
        return res
    

if __name__ == '__main__':
    model = VietOCRPrediction(
        config_name= 'vgg_transformer',
        weight_path= './media/icnlab/Data/Manh/OCR/Documentation/Text_Recognize/weights/transformerocr.pth', 
    )
    img = Image.open('/media/icnlab/Data/Manh/OCR/Documentation/Text_Recognize/invoice_ocr/img_crop/X00016469612_0.jpg')
    res = model.predict(img)
    print(res, type(res))




