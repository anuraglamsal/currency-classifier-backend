from flask import Flask, request, send_file
import os
from scripts import modelScript, ttsScript

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def handleRequest():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = file.filename
        filepath = os.path.join('uploads/', filename)
        file.save(filepath)

    label = modelScript.predict(file.filename)

    os.remove(f'uploads/{file.filename}')

    lang = "nep"

    ttsScript.generateAudio(label, lang)

    return send_file('uploads/output.mp3', as_attachment=True)

if __name__ == '__main__':
    app.run(host='192.168.1.70', port=3000)

