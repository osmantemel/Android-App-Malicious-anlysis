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
    def __init__(self, data_path,secilen_ozellikler, secilen_degerler):
        self.TumSonuclar = {}
        self.veriseti_ayarla(data_path,secilen_ozellikler, secilen_degerler)
       
    def veriseti_ayarla(self, data_path,secilen_ozellikler, secilen_degerler):
        self.data = pd.read_csv(data_path)
        self.data = self.data.drop('TelephonyManager.getSimCountryIso', axis=1)
        self.y = self.data["class"]
        self.X = self.data[secilen_ozellikler]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def _model_egit(self, model, model_ad,secilen_degerler):
        
        secilen_degerler_df = pd.DataFrame(secilen_degerler)    
        secilen_degerler_df=secilen_degerler_df.T  
        secilen_degerler_df = secilen_degerler_df.fillna(0)
        current_rows, current_cols = secilen_degerler_df.shape
        target_rows = 3008
        target_cols = current_cols
        if target_rows > current_rows:
            num_rows_to_add = target_rows - current_rows
            extension_df = pd.DataFrame(0, index=range(num_rows_to_add), columns=secilen_degerler_df.columns)
            secilen_degerler_df = pd.concat([secilen_degerler_df, extension_df])
        
        secilen_degerler_df.columns = self.X_test.columns
        secilen_degerler_df.index = self.X_test.index

        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.TumSonuclar[model_ad] = accuracy
        y_pred_gercek = model.predict(secilen_degerler_df)
        return y_pred_gercek
    
    def model_LR(self,secilen_degerler):
        model_LR = LogisticRegression()
        y_pred=self._model_egit(model_LR, "model_LR",secilen_degerler)
        return y_pred

    def model_SVM(self,secilen_degerler):
        model_SVM = SVC()
        y_pred=self._model_egit(model_SVM, "model_SVM",secilen_degerler)
        return y_pred

    def model_DT(self,secilen_degerler):
        model_DT = DecisionTreeClassifier()
        y_pred=self._model_egit(model_DT, "model_DT",secilen_degerler)
        return y_pred
        

    def model_RF(self,secilen_degerler):
        model_RF = RandomForestClassifier()
        y_pred=self._model_egit(model_RF, "model_RF",secilen_degerler)
        return y_pred
        

    def model_MLP(self,secilen_degerler):
        model_MLP = MLPClassifier()
        y_pred=self._model_egit(model_MLP, "model_MLP",secilen_degerler)
        return y_pred
        

    def model_GNB(self,secilen_degerler):
        model_GNB = GaussianNB()
        y_pred=self._model_egit(model_GNB, "model_GNB",secilen_degerler)
        return y_pred
        

    def en_iyi_model_ve_accuracy(self):
        en_iyi_model_ad = max(self.TumSonuclar, key=self.TumSonuclar.get)
        en_iyi_model_accuracy = self.TumSonuclar[en_iyi_model_ad]
        return en_iyi_model_ad, en_iyi_model_accuracy

