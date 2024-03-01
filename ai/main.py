from model import AiEgitimSonuc

ai_egitim_sonucu = AiEgitimSonuc('/home/osman/Documents/projeler/Android-App-Malicious-anlysis/datasets/drebin-215-dataset-5560malware-9476-benign.csv')

ai_egitim_sonucu.model_LR()
ai_egitim_sonucu.model_DT()
ai_egitim_sonucu.model_RF()
ai_egitim_sonucu.model_MLP()
ai_egitim_sonucu.model_GNB()
ai_egitim_sonucu.model_SVM()

en_iyi_model, en_iyi_model_accuracy = ai_egitim_sonucu.en_iyi_model_ve_accuracy()
print(f"En iyi model: {en_iyi_model}, Accuracy: {en_iyi_model_accuracy}")

def yapayZeka(secilen_ozellikler,secilen_degerler):
    return "osamn"
    