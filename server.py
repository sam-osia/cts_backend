import scipy.ndimage
from flask import Flask
from flask import request, send_file
from flask_cors import CORS
import datetime
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time

from matplotlib import rcParams

from image_processing import filter_mask, standardize_projection
import pandas as pd
import json
import imutils
from zipfile import ZipFile
import sys
from model import load_model


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
viridis = plt.get_cmap(rcParams["image.cmap"])

app = Flask(__name__)
cors = CORS(app)
current_capture_method = None
current_subject_id = None
current_model_input = None
subject_images = None
selected_image = None
patients_list = None


def load_subject_images(capture_method, subject_id):
    print('loading subject images')
    global subject_images
    if capture_method == 'android':
        image_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android/{subject_id}/top_parsed'
        files = os.listdir(image_parent_dir)
        num_frames = len([file for file in files if file.startswith('confidence')]) - 1
        subject_images = [cv2.imread(os.path.join(image_parent_dir, f'depth_{i}.png'), -1) for i in range(num_frames)]
    elif capture_method == 'iphone':
        subject_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/iPhone/Top_View/Iphone_top{subject_id}'
        depths_dir = [dir for dir in os.listdir(subject_parent_dir) if dir.startswith('depth')][0]
        image_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/iPhone/Top_View/Iphone_top{subject_id}/{depths_dir}'
        files = os.listdir(image_parent_dir)
        num_frames = len(files)
        subject_images = [cv2.imread(os.path.join(image_parent_dir, f'{i}.exr'), -1) for i in range(num_frames)]


def convert_to_rgb(frame: np.ndarray):
    max = frame.max(initial=0)
    min = frame.min(initial=255)
    return np.interp(frame, [min, max], [0, 255]).astype(np.uint8)


# @app.route("/allPatients")
def get_patients():
    patient_capture_dir = '/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android'
    patients = os.listdir(patient_capture_dir)

    return {
        'patients': [
            {
                'MRN': i,
                'BirthDate': random.randint(2, 12),
                'Diagnosis': random.choice(['Sagittal', 'Metopic', 'Unicoronal', 'Plagiocephaly', 'Normal', 'Other']),
                'AndroidID': patient,
                'iPhoneID': f'i{patient}'
            }
            for i, patient in enumerate(patients)],
    }


@app.route("/allPatients")
def get_patients_from_log():
    print('here')
    df = pd.read_excel('/hpf/largeprojects/dsingh/cts/data_transfers/Metadata/23-04-03-Combined3D-LOG.ods',
                       engine='odf')
    df_captured = df.copy(deep=True)
    df_captured.dropna(subset=['Android ID', 'Iphone ID (Top, Back, Sides)'], how='all', inplace=True)
    patients = []
    for i, row in df_captured.iterrows():
        patient = {
            'MRN': int(row['MRN']),
            'BirthDate': str(row['Birth date']),
            'Diagnosis': str(row['Dx Synostosis']),
            'AndroidID': str(row['Android ID']),
            'iPhoneID': str(row['Iphone ID (Top, Back, Sides)']).split(',')[0][10:]
        }
        patients.append(patient)

    return {'patients': patients}


@app.route('/maxFrameNum/<capture_method>/<subject_id>')
def get_max_frame_num(capture_method=None, subject_id=None):
    if capture_method == 'android':
        image_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android/{subject_id}/top_parsed'
        files = os.listdir(image_parent_dir)
        files = [file for file in files if file.startswith('confidence')]
    elif capture_method == 'iphone':
        subject_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/iPhone/Top_View/Iphone_top{subject_id}'
        depths_dir = [dir for dir in os.listdir(subject_parent_dir) if dir.startswith('depth')][0]
        image_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/iPhone/Top_View/Iphone_top{subject_id}/{depths_dir}'
        files = os.listdir(image_parent_dir)
    else:
        raise Exception('Invalid capture method')

    return {
        'maxFrameNum': len(files) - 2,
    }


@app.route("/image", methods=['POST'])
@app.route("/image/<capture_method>/<subject_id>", methods=['POST'])
def get_image_filepath(capture_method=None, subject_id=None):
    global current_capture_method
    global current_subject_id
    global current_model_input
    print(request.get_json())

    if subject_id is None:
        subject_id = request.get_json()['subjectId']

    if capture_method is None:
        capture_method = request.get_json()['captureMethod']

    if subject_id != current_subject_id or capture_method != current_capture_method:
        load_subject_images(capture_method, subject_id)
        current_capture_method = capture_method
        current_subject_id = subject_id

    print(request.get_json())
    frame_num = request.get_json()['frameNum']
    threshold = request.get_json()['depthThreshold']
    rotation = request.get_json()['rotation']
    use_edge_disntace = request.get_json()['useEdgeDistance']
    min_edge_distance = request.get_json()['minEdgeDistance']

    selected_image = subject_images[frame_num].copy()
    # plt.subplot(121), plt.imshow(selected_image)

    if capture_method == 'iphone':
        selected_image = selected_image[:, :, 2] * 5000
        # change type to int
        selected_image = selected_image.astype(np.uint16)
        selected_image = cv2.rotate(selected_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        selected_image = cv2.resize(selected_image, fx=2.5, fy=2.5, dsize=(0, 0), interpolation=cv2.INTER_NEAREST)

    selected_image[selected_image > threshold] = 0

    if use_edge_disntace:
        try:
            image_mask = filter_mask(selected_image, minDistEdge=min_edge_distance)
            selected_image = cv2.bitwise_and(selected_image, selected_image, mask=image_mask)
        except Exception as e:
            print(e)

    rotated_selected_image = imutils.rotate(convert_to_rgb(selected_image), rotation)
    cv2.imwrite('./resources/depth_rgb.png', rotated_selected_image)

    # for standardizing projection
    processed_image = standardize_projection(selected_image, invert=True)
    if rotation != 0:
        processed_image = imutils.rotate_bound(processed_image, rotation)
        # do it again to fix the padded square in case of rotation
        processed_image = standardize_projection(processed_image)

    current_model_input = processed_image.copy()

    processed_image = convert_to_rgb(processed_image)
    processed_image = viridis(processed_image)[..., :3]
    processed_image = (processed_image * 255).astype(np.uint8)

    cv2.imwrite('./resources/processed.png', processed_image)
    files = ['depth_rgb.png', 'processed.png']
    with ZipFile('./resources/files.zip', 'w') as zipObj:
        for file in files:
            zipObj.write(os.path.join('./resources', file), file)

    return send_file('./resources/files.zip', mimetype='zip', as_attachment=True)


@app.route("/predict", methods=['GET'])
def predict():
    global current_model_input
    image_for_prediction = np.reshape(current_model_input, (1, 128, 128, 1))
    result = model.predict(image_for_prediction)[0]
    # set aspect ratio to be a square
    fig = plt.figure(figsize=(10, 10))
    plt.pie(result, labels=['Sagital', 'Metopic', 'Unicoronal', 'Plagiocephaly', 'Normal'])
    plt.savefig('./resources/prediction.png')
    return send_file('./resources/prediction.png', mimetype='image/png')


@app.route("/submitImage/<capture_method>/<subject_id>", methods=['POST'])
def submit_image(capture_method=None, subject_id=None):
    file_path = get_file_path(capture_method, subject_id)
    print(request.get_json())
    print(type(request.get_json()))
    print(os.path.join(file_path, 'preprocessing_info.json'))
    with open(os.path.join(file_path, 'preprocessing_info.json'), 'w') as f:
        json.dump(request.get_json(), f, ensure_ascii=False, indent=4)
    return {}


@app.route("/addPatient", methods=['POST'])
def add_patient():
    print(request.get_json())
    return {
        'message': 'Patient added successfully',
    }


def get_file_path(capture_method, subject_id):
    if capture_method == 'android':
        image_path = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android/{subject_id}/top_parsed'
    elif capture_method == 'iphone':
        image_path = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/iPhone/Top_View/Iphone_top{subject_id}'
    else:
        raise Exception('Invalid capture method')

    return image_path


if __name__ == "__main__":
    model = load_model()
    app.run(debug=True)
    # get_patients_from_log()
