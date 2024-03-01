from flask import Flask, request, jsonify
from flask_cors import CORS
from model import AiEgitimSonuc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import json
app = Flask(__name__)
CORS(app)
TumSonuclar1 = []

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        data = request.json
        secilen_ozellikler = data.get('secilenOzellikler', [])
        secilen_degerler = data.get('secilenDegerler', [])
        
        data_path = '/home/osman/Documents/projeler/Android-App-Malicious-anlysis/datasets/drebin-215-dataset-5560malware-9476-benign.csv'
        yapayZeka(secilen_ozellikler, secilen_degerler, data_path)
        print(type(TumSonuclar1))
 

        return jsonify({'response':"TumSonuclar1"})
    except Exception as e:
        print(f'Hata: {str(e)}')
        return jsonify({'error': f'Hata: {str(e)}'}), 500

def yapayZeka(secilen_ozellikler, secilen_degerler, data_path):
    secilen_degerler = pd.DataFrame(secilen_degerler)

    ai_egitim_sonucu = AiEgitimSonuc(data_path,secilen_ozellikler, secilen_degerler)
    ai_egitim_sonucu.model_LR(secilen_degerler)
    ai_egitim_sonucu.model_DT(secilen_degerler)
    ai_egitim_sonucu.model_RF(secilen_degerler)
    ai_egitim_sonucu.model_MLP(secilen_degerler)
    ai_egitim_sonucu.model_GNB(secilen_degerler)
    ai_egitim_sonucu.model_SVM(secilen_degerler)

    en_iyi_model, en_iyi_model_accuracy = ai_egitim_sonucu.en_iyi_model_ve_accuracy()

    print(f"En iyi model: {en_iyi_model}, Accuracy: {en_iyi_model_accuracy}")
    
    if en_iyi_model == "model_LR":
        y_pred=ai_egitim_sonucu.model_LR(secilen_degerler)
    elif en_iyi_model == "model_DT":
        y_pred=ai_egitim_sonucu.model_DT(secilen_degerler)
    elif en_iyi_model == "model_RF":
        y_pred=ai_egitim_sonucu.model_RF(secilen_degerler)
    elif en_iyi_model == "model_MLP":
        y_pred=ai_egitim_sonucu.model_MLP(secilen_degerler)
    elif en_iyi_model == "model_GNB":
        y_pred=ai_egitim_sonucu.model_GNB(secilen_degerler)
    elif en_iyi_model == "model_SVM":
        y_pred=ai_egitim_sonucu.model_SVM(secilen_degerler)  
    else:
        print("MODEL TESİNTE HATA OLUŞTU !!")
    TumSonuclar1.append(en_iyi_model)
    TumSonuclar1.append(en_iyi_model_accuracy)
    TumSonuclar1.append(y_pred[0:len(secilen_degerler)])
    print(TumSonuclar1)


if __name__ == '__main__':
    app.run(debug=True)
