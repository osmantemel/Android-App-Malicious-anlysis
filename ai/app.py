from flask import Flask, request, jsonify
from flask_cors import CORS
from model import AiEgitimSonuc
import pandas as pd
app = Flask(__name__)
CORS(app)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        data = request.json
        secilen_ozellikler = data.get('secilenOzellikler', [])
        secilen_degerler = data.get('secilenDegerler', [])
        
        data_path = '/home/osman/Documents/projeler/Android-App-Malicious-anlysis/datasets/drebin-215-dataset-5560malware-9476-benign.csv'
        yapayZeka(secilen_ozellikler, secilen_degerler, data_path)

        return jsonify({'response': 'Veri başarıyla alındı'})
    except Exception as e:
        print(f'Hata: {str(e)}')
        return jsonify({'error': f'Hata: {str(e)}'}), 500

def yapayZeka(secilen_ozellikler, secilen_degerler, data_path):
    ai_egitim_sonucu = AiEgitimSonuc(data_path,secilen_ozellikler, secilen_degerler)
    ai_egitim_sonucu.model_LR()
    ai_egitim_sonucu.model_DT()
    ai_egitim_sonucu.model_RF()
    ai_egitim_sonucu.model_MLP()
    ai_egitim_sonucu.model_GNB()
    ai_egitim_sonucu.model_SVM()

    en_iyi_model, en_iyi_model_accuracy = ai_egitim_sonucu.en_iyi_model_ve_accuracy()
    
    
    print(f"En iyi model: {en_iyi_model}, Accuracy: {en_iyi_model_accuracy}")
    
    if en_iyi_model == "model_LR":
        secilen_degerler  = pd.DataFrame(secilen_degerler)
        print(secilen_degerler)
    elif en_iyi_model == "model_DT":
        print(secilen_degerler)
        
    elif en_iyi_model == "model_RF":
        print(secilen_degerler)

    elif en_iyi_model == "model_MLP":
        print(secilen_degerler)

    elif en_iyi_model == "model_GNB":
        print(secilen_degerler)
    elif en_iyi_model == "model_SVM":
        print(secilen_degerler)
    else:
        print("ye beni osman")
        
    

if __name__ == '__main__':
    app.run(debug=True)
