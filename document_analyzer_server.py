from flask import Flask, request, jsonify
from flask_cors import CORS 
import uuid
import os
import json
from document_analyze import upload_file, async_detect_document, check_results, \
    write_to_text, translate_text, delete_objects, run_chatgpt_api,\
    ask_chatgpt_question

from constants import GOOGLE_CLOUD_BUCKET


app = Flask(__name__)
CORS(app)
BUCKET = GOOGLE_CLOUD_BUCKET


@app.route('/')
def hello_world():
    return "Hello from flask!"

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    file_to_analyze = request.files['file']
    batch_name = str(uuid.uuid4())
    local_file_path = 'uploads/%s.pdf' % batch_name
    cloud_file_path = '%s.pdf' % batch_name
    file_to_analyze.save(local_file_path)
    upload_file(local_file_path, BUCKET, cloud_file_path)
    async_detect_document(
        f'gs://{BUCKET}/{cloud_file_path}',
        f'gs://{BUCKET}/{batch_name}')

    return jsonify({
        'batchId': batch_name
    })

@app.route('/check_if_finished', methods=['POST'])
def check_if_finished():
    batch_name = request.json['batchId']
    if not check_results(BUCKET, batch_name):
        return jsonify({
            'status': 'processing'
        })
    write_to_text(BUCKET, batch_name)
    all_responses = []
    for result_json in os.listdir('ocr_results'):
        if result_json.endswith('json') and result_json.startswith(batch_name):
            result_file = os.path.join('ocr_results', result_json)
            with open(os.path.join('ocr_results', result_json)) as fp_res:
                response = json.load(fp_res)
            all_responses.extend(response['responses'])
            os.remove(result_file)
    txts = [a['fullTextAnnotation']['text'] for a in all_responses]
    translated_text = [translate_text(t) for t in txts]
    print('Running cleanup...')
    delete_objects(BUCKET, batch_name)
    os.remove('uploads/%s.pdf' % batch_name)
    analysis = run_chatgpt_api('\n'.join(translated_text))
    analysis_res = json.loads(analysis)
    return jsonify({
        'status': 'complete',
        'analysis': analysis,
        'translatedText': translated_text,
        'rawText': '\n'.join(txts)
    })

@app.route('/ask_user_question', methods=['POST'])
def ask_user_question():
    report_text = request.json['text']
    user_question = request.json['userQuestion']
    response = ask_chatgpt_question(report_text, user_question)
    return jsonify({
        'result': response
    })

if __name__ == '__main__':
    app.run(debug=True)
