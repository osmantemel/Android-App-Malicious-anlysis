from flask import Flask, request, jsonify
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        data = request.json
        secilen_ozellikler = data.get('secilenOzellikler', [])
        secilen_degerler = data.get('secilenDegerler', [])
        
        yapayZeka(secilen_ozellikler,secilen_degerler)

        return jsonify({'response': 'Veri başarıyla alındı'})
    except Exception as e:
        print(f'Hata: {str(e)}')
        return jsonify({'error': f'Hata: {str(e)}'}), 500

def yapayZeka(secilen_ozellikler,secilen_degerler):
    print(secilen_ozellikler)
    print(secilen_degerler)

if __name__ == '__main__':
    app.run(debug=True)


