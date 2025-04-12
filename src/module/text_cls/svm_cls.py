import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
def tfidf(copus_train, copus_test):
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(copus_train)
    copus_train = vectorizer.transform(copus_train)
    copus_test = vectorizer.transform(copus_test)
    return copus_train, copus_test,vectorizer

def svm_cls(copus_train, label_train, copus_test, label_test):
    svc = SVC(random_state=42)
    svc.fit(copus_train, label_train)
    y_pred = svc(copus_test)
    print(classification_report(label_test, y_pred))
    return svc
class SVMClassifier():
    def __init__(self, tfidf_path, svm_path):
        try:
            with open(tfidf_path, 'rb') as f:
                self.tfidf_model = pickle.load(f)
            with open(svm_path, 'rb') as f:
                self.svm_model = pickle.load(f)
        except Exception as e:
            print(f'Error when reading model file. Error: {e}')
    def predict(self, text):
        text = [text]
        encoded_text = self.tfidf_model.transform(text)
        res_pred = self.svm_model.predict(encoded_text)
        return res_pred
        

# if __name__ == '__main__':
#     tfidf_path = 'weights/model_vectorizer_tfidf.pkl'
#     svm_path = 'weights/svm_classifier_model.pkl'
#     svm =  SVMClassifier(tfidf_path=tfidf_path,  svm_path=svm_path)
#     text = 'Tổng số tiền:'
#     print(svm.predict(text))


