class PhoBertPrediction():
    def __init__(self,weight_path, num_cls = 3, max_len = 256):
        self.model = BERTClassifier(num_cls)
        self.rdrsegmenter = VnCoreNLP(
            "/media/icnlab/Data/Manh/OCR/Documentation/Text_Classification/weights/VnCoreNLP-1.2.jar", 
            annotators="wseg", 
            max_heap_size='-Xmx500m'
        )
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.max_len = max_len

        model_state_dict = torch.load(weight_path, weights_only=True)['model_state_dict']
        self.model.load_state_dict(model_state_dict)
    def predict(self, text):
        self.model.eval()
        text_processed = self.process_text(text)
        input_ids, input_mask = torch.tensor(text_processed['input_ids']), torch.tensor(text_processed['attention_mask'])
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask = input_mask,
                labels = None
            )
            logits = outputs.logits.detach().cpu().numpy()
            predict_flat = np.argmax(logits, axis = 1).flatten()
        return predict_flat
    def process_text(self, text):
        text_tokenized = self.tokenize(text)
        inputs = self.tokenizer(
            text_tokenized,
            padding = 'max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'],  # Remove batch dimension
            'attention_mask': inputs['attention_mask']
        }
    def tokenize(self, text):
        try:
            sents = self.rdrsegmenter.tokenize(text)
            text_tokenized = ' '.join([' '.join(sent) for sent in sents])
        except Exception as e:
            print(f'Failed to tokenize text: {text}. Error: {e}')
            text_tokenized = ''
        return text_tokenized