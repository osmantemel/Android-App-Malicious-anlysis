import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class AiEgitimSonuc:
    def __init__(self, data_path):
        self.TumSonuclar = {}
        self.data = pd.read_csv(data_path)
        self.data = self.data.drop('TelephonyManager.getSimCountryIso', axis=1)
        self.y = self.data["class"]
        self.X = self.data.drop("class", axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def _model_egit(self, model, model_ad):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.TumSonuclar[model_ad] = accuracy

    def model_LR(self):
        model_LR = LogisticRegression()
        self._model_egit(model_LR, "model_LR")

    def model_SVM(self):
        model_SVM = SVC()
        self._model_egit(model_SVM, "model_SVM")

    def model_DT(self):
        model_DT = DecisionTreeClassifier()
        self._model_egit(model_DT, "model_DT")

    def model_RF(self):
        model_RF = RandomForestClassifier()
        self._model_egit(model_RF, "model_RF")

    def model_MLP(self):
        model_MLP = MLPClassifier()
        self._model_egit(model_MLP, "model_MLP")

    def model_GNB(self):
        model_GNB = GaussianNB()
        self._model_egit(model_GNB, "model_GNB")

    def en_iyi_model_ve_accuracy(self):
        en_iyi_model_ad = max(self.TumSonuclar, key=self.TumSonuclar.get)
        en_iyi_model_accuracy = self.TumSonuclar[en_iyi_model_ad]
        return en_iyi_model_ad, en_iyi_model_accuracy


