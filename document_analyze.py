import requests
import openai
from google.cloud import vision
from google.cloud import storage
import os
import json
import time
import shutil
from constants import OPENAPI_KEY, GOOGLE_CREDENTIALS_PATH, GOOGLE_TRANSLATE_KEY, GOOGLE_CLOUD_BUCKET

openai.api_key = OPENAPI_KEY
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_CREDENTIALS_PATH

# OCR

# upload the file to google cloud storage bucket for further processing
def upload_file(file_name, bucket_name, bucket_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(file_name)

# kick off the OCR process on the document. This is asynchronous because OCR can take a while
def async_detect_document(gcs_source_uri, gcs_destination_uri):
    client = vision.ImageAnnotatorClient()
    input_config = vision.InputConfig(gcs_source=vision.GcsSource(uri=gcs_source_uri), mime_type= 'application/pdf')
    output_config = vision.OutputConfig(
       gcs_destination=vision.GcsDestination(uri=gcs_destination_uri), 
       batch_size=100
    )
    async_request = vision.AsyncAnnotateFileRequest(
        features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)], 
        input_config=input_config, output_config=output_config
    )
    operation = client.async_batch_annotate_files(requests=[async_request])

def check_results(bucket_path, prefix):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_path)
    blob_list = list(bucket.list_blobs(prefix=prefix))
    blb = [b for b in blob_list if 'output-' in b.name and '.json' in b.name]
    return len(blb) != 0

def delete_objects(bucket, prefix):
    bucket = storage.Client().get_bucket(bucket)
    blob_list = list(bucket.list_blobs(prefix=prefix))
    for blob in blob_list:
        blob.delete()
        print('Blob', blob.name, 'Deleted')


# download the OCR result file
def write_to_text(bucket_name, prefix):
    bucket = storage.Client().get_bucket(bucket_name)
    blob_list = list(bucket.list_blobs(prefix=prefix))
    if not os.path.exists('ocr_results'):
        os.mkdir('ocr_results')
    for blob in blob_list:
        if blob.name.endswith('.json'):
            with open(os.path.join('ocr_results', blob.name), 'w') as fp_data:
                print(blob.download_as_string())
                fp_data.write(blob.download_as_string().decode('utf-8'))



# Translation

# detect the language we’re translating from
def detect_language(text):
    url = 'https://translation.googleapis.com/language/translate/v2/detect'
    data = {
        "q": text,
        "key": GOOGLE_TRANSLATE_KEY
    }
    res = requests.post(url, data=data)
    return res.json()['data']['detections'][0][0]['language']

# translate the text
def translate_text(text):
    url = 'https://translation.googleapis.com/language/translate/v2'
    language = detect_language(text)
    if language == 'en':
        return text
    data = {
        "q": text,
        "source": language,
        "target": "en",
        "format": "text",
        "key": GOOGLE_TRANSLATE_KEY
    }
    res = requests.post(url, data=data)
    return res.json()['data']['translations'][0]['translatedText']

# ChatGPT and Prompt Engineering


# run the first pass analysis on the text using ChatGPT
def run_chatgpt_api(report_text):
    completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": '''
        Consider the following report:
        ---
        %s
        ---
        1. Summarize the purpose of the report.
        2. Summarize the primary conclusion of the report.
        3. Summarize the secondary conclusion of the report
        4. Who is the intended audience for this report?
        5. What other additional context would a reader be interested in knowing?
Please reply in json format with the keys purpose, primaryConclusion, secondaryConclusion, intendedAudience, and additionalContextString.
            ''' % report_text},
          ]
        )
    return completion.choices[0]['message']['content']

def ask_chatgpt_question(report_text, question_text):
    completion = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "user", "content": '''
        Consider the following report:
        ---
        %s
        ---
        Answer the following question:
        %s
            ''' % (report_text, question_text)},
          ]
        )
    return completion.choices[0]['message']['content']

if __name__ == '__main__':
    bucket = GOOGLE_CLOUD_BUCKET
    upload_file_name = 'govtreport.pdf'
    upload_prefix = 'govtreport_analyzed'

    upload_file('Japan_Mext.pdf', bucket, upload_file_name)
    async_detect_document(f'gs://{bucket}/{upload_file_name}', f'gs://{bucket}/{upload_prefix}')
    while not check_results(bucket, upload_prefix):
        print('Not done yet... checking again')
        time.sleep(5)
        

    write_to_text(bucket, upload_prefix)
    all_responses = []
    for result_json in os.listdir('ocr_results'):
        with open(os.path.join('ocr_results', result_json)) as fp_res:
            response = json.load(fp_res)
        all_responses.extend(response['responses'])
    txts = [a['fullTextAnnotation']['text'] for a in all_responses]

    translated_text = [translate_text(t) for t in txts]
    print('Running cleanup...')
    delete_objects(bucket, upload_file_name)
    delete_objects(bucket, upload_prefix)
    shutil.rmtree('ocr_results')

    print('Running Analysis...')
    analysis = run_chatgpt_api('\n'.join(translated_text))
    analysis_res = json.loads(analysis)

    print('=== purpose ====')
    print(analysis_res['purpose'])
    print()
    print('==== primary conclusion =====')
    print(analysis_res['primaryConclusion'])
    print()
    print('==== secondary conclusion =====')
    print(analysis_res['secondaryConclusion'])
    print()
    print('==== intended audience ====')
    print(analysis_res['intendedAudience'])
    print()
    print('===== additional context =====')
    print(analysis_res['additionalContextString'])
