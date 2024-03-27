from flask import Flask, request, send_file, after_this_request
from scripts import modelScript, ttsScript, segScript
import os, time

app = Flask(__name__)

@app.route('/audio', methods=['POST'])
def audioRequest():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    lang_idx = int(request.form['lang_idx'])

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = file.filename
        filepath = os.path.join('uploads/', filename)
        file.save(filepath)

    label = modelScript.predict(file.filename)

    segScript.bounded_image(file.filename)

    os.remove(f'uploads/{file.filename}')

    lang = "nep" if not lang_idx else "eng"

    ttsScript.generateAudio(label, lang)

    return send_file('uploads/output.mp3', as_attachment=True, etag=label)

@app.route('/image', methods=['GET'])
def imageRequest():

    @after_this_request
    def cleanup(response):
        os.remove('bounded_image.png')
        return response

    while not os.path.exists('bounded_image.png'):
        time.sleep(1)

    time.sleep(1)

    return send_file('bounded_image.png', as_attachment=True)

if __name__ == '__main__':
    app.run(host='192.168.1.70', port=3000)
