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
from image_processing import filter_mask


app = Flask(__name__)
cors = CORS(app)
current_subject_id = None
subject_images = None
selected_image = None
patients_list = None


def load_subject_images(subject_id):
    global subject_images
    image_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android/{subject_id}/top_parsed'
    files = os.listdir(image_parent_dir)
    num_frames = len([file for file in files if file.startswith('confidence')]) - 1
    subject_images = [cv2.imread(os.path.join(image_parent_dir, f'depth_{i}.png'), -1) for i in range(num_frames)]


def convert_to_rgb(frame: np.ndarray):
    max = frame.max(initial=0)
    min = frame.min(initial=255)
    return np.interp(frame, [min, max], [0, 255]).astype(np.uint8)


@app.route("/allPatients")
def get_patients():
    patient_capture_dir = '/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android'
    patients = os.listdir(patient_capture_dir)
    print('hello')

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


@app.route('/maxFrameNum/<subject_id>')
def get_max_frame_num(subject_id=None):
    image_parent_dir = f'/hpf/largeprojects/dsingh/cts/data_transfers/uploaded_scans/Android/{subject_id}/top_parsed'
    files = os.listdir(image_parent_dir)
    files = [file for file in files if file.startswith('confidence')]
    return {
        'maxFrameNum': len(files) - 2,
    }


@app.route("/image", methods=['POST'])
@app.route("/image/<subject_id>", methods=['POST'])
def get_image_filepath(subject_id=None):
    global current_subject_id

    if subject_id is None:
        subject_id = request.get_json()['subjectId']

    if subject_id != current_subject_id:
        load_subject_images(subject_id)
        current_subject_id = subject_id

    print(request.get_json())
    frame_num = request.get_json()['frameNum']
    threshold = request.get_json()['depthThreshold']
    use_edge_disntace = request.get_json()['useEdgeDistance']
    min_edge_distance = request.get_json()['minEdgeDistance']

    selected_image = subject_images[frame_num].copy()
    selected_image[selected_image > threshold] = 0

    if use_edge_disntace:
        print('here')
        try:
            image_mask = filter_mask(selected_image, minDistEdge=min_edge_distance)
            selected_image = cv2.bitwise_and(selected_image, selected_image, mask=image_mask)
        except Exception as e:
            print(e)

    selected_image = convert_to_rgb(selected_image)
    cv2.imwrite('./resources/depth_rgb.png', selected_image)
    return send_file('./resources/depth_rgb.png', mimetype='image/png')


@app.route("/addPatient", methods=['POST'])
def add_patient():
    print(request.get_json())
    return {
        'message': 'Patient added successfully',
    }


if __name__ == "__main__":
    app.run(debug=True)
