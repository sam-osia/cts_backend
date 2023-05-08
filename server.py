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

from image_processing import filter_mask, standardize_projection, binvox_to_projection, get_cephalic_index
import pandas as pd
import json
import imutils
from zipfile import ZipFile
import sys

import sqlite3
import time
from tqdm import tqdm


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


@app.route("/allPatients")
def get_patients_from_log():
    global patients_list

    if patients_list is None:
        print('Retrieving patients from log')
        conn = sqlite3.connect('/hpf/largeprojects/dsingh/cts/squishy.sql')
        sql = "SELECT * from subjects join images on subjects.subject_id = images.subject_id where num_dimensions = 3 and labels_id = 3"
        df_sql = pd.read_sql_query(sql, conn)
        df_sql = df_sql.loc[:, ~df_sql.columns.duplicated()]
        df_sql.mrn = df_sql.mrn.astype(int, errors='ignore')

        df = pd.read_excel('/hpf/largeprojects/dsingh/cts/data_transfers/Metadata/MasterLog/MasterLog.ods',
                           engine='odf')
        df_captured = df.copy(deep=True)
        df_captured.dropna(subset=['Android ID', 'Iphone ID (Top, Back, Sides)'], how='all', inplace=True)
        df_captured.MRN = df_captured.MRN.astype(int, errors='ignore')
        patients = []

        for i, row in (df_captured.iterrows()):
            iphone_id = str(row['Iphone ID (Top, Back, Sides)']).split(',')[0][10:]
            iphone_path = get_file_path('iphone', iphone_id)
            android_id = str(row['Android ID'])
            android_path = get_file_path('android', android_id)

            iphone_processed = os.path.exists(os.path.join(iphone_path, 'preprocessing_info.json'))
            android_processed = os.path.exists(os.path.join(android_path, 'preprocessing_info.json'))
            binvox_path = None

            patient_sql = df_sql[df_sql['mrn'] == row['MRN']]
            if len(patient_sql) > 0:
                patient_sql = patient_sql.iloc[0]
                md3d_parent_path = os.path.join('/hpf/largeprojects/dsingh/cts/raw_data', patient_sql['folder'])
                for file in os.listdir(md3d_parent_path):
                    if file.endswith('edited.binvox'):
                        binvox_path = os.path.join(md3d_parent_path, file)
                        break
            else:
                print(f'Patient {row["MRN"]} not found in SQL')

            patient = {
                'MRN': row['MRN'],
                'BirthDate': str(row['Birth date'])[:10],
                'Diagnosis': str(row['Dx Synostosis']),
                'AndroidID': android_id,
                'iPhoneID': iphone_id,
                'AndroidProcessed': android_processed,
                'iPhoneProcessed': iphone_processed,
                '3dmdPath': binvox_path,
            }
            patients.append(patient)
        patients_list = patients
    else:
        print('Retrieving patients from cache')

    return {'patients': patients_list}


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


@app.route('/properties/<capture_method>/<subject_id>')
def get_properties(capture_method=None, subject_id=None):
    default_properties = {
        'frameNum': 0,
        'depthThreshold': 1000,
        'rotation': 0,
        'useEdgeDistance': False,
        'minEdgeDistance': 1000,
        'minArea': 1000,
    }

    image_parent_dir = get_file_path(capture_method, subject_id)
    properties_path = os.path.join(image_parent_dir, 'preprocessing_info.json')
    if os.path.exists(properties_path):
        with open(properties_path, 'r') as f:
            properties = json.load(f)
    else:
        properties = default_properties

    return properties

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
        processed_image = imutils.rotate(processed_image, rotation)
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
    fig = plt.figure(figsize=(20, 20))
    labels = ['Sagital', 'Metopic', 'Unicoronal', 'Plagiocephaly', 'Normal']
    legend_text = [f'{label}: {round(result[i] * 100, 2)}%' for i, label in enumerate(labels)]
    patches, texts = plt.pie(result)
    plt.legend(patches, legend_text, loc="best", prop={'size': 36})
    plt.savefig('./resources/prediction.png', dpi=300)
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


@app.route("/compareScans/<patientMRN>", methods=['POST'])
def get_image_comparisons(patientMRN=None):
    mode = request.get_json()['imageMode']
    display_cephalic_index = request.get_json()['displayCephalicIndex']
    patient_info = next((item for item in patients_list if item["MRN"] == int(patientMRN)), None)
    print(patient_info)

    has_3dmd = False
    has_android = False
    has_iphone = False

    if patient_info['3dmdPath'] is not None:
        projection = binvox_to_projection(patient_info['3dmdPath'])

        if display_cephalic_index:
            (left_edge, max_width_position), (right_edge, max_width_position), \
                (max_height_position, top_edge), (max_height_position, bottom_edge) = get_cephalic_index(projection)
            cephalic_index = (left_edge - right_edge) / (top_edge - bottom_edge)
            print('3dmd cephalic index: ', cephalic_index)

        projection = convert_to_rgb(projection)
        projection = viridis(projection)
        projection = (projection * 255).astype(np.uint8)

        if display_cephalic_index:
            projection = cv2.circle(projection, (left_edge, max_width_position), 2, (0, 0, 255), -1)
            projection = cv2.circle(projection, (right_edge, max_width_position), 2, (0, 0, 255), -1)
            projection = cv2.circle(projection, (max_height_position, top_edge), 2, (0, 0, 255), -1)
            projection = cv2.circle(projection, (max_height_position, bottom_edge), 2, (0, 0, 255), -1)

            projection = cv2.line(projection, (left_edge, max_width_position), (right_edge, max_width_position), (0, 0, 255), 1)
            projection = cv2.line(projection, (max_height_position, top_edge), (max_height_position, bottom_edge), (0, 0, 255), 1)
            # projection = cv2.putText(projection, f'Cephalic Index: {round(cephalic_index, 2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255))

        cv2.imwrite('./resources/3dmd_compare.png', projection)
        has_3dmd = True
    if patient_info['AndroidID'] != '':
        android_image_path = get_file_path('android', patient_info['AndroidID'])
        if 'preprocessing_info.json' in os.listdir(android_image_path):
            with open(os.path.join(android_image_path, 'preprocessing_info.json')) as f:
                preprocessing_info = json.load(f)
                print('android preprocessing info')
                print(preprocessing_info)

            frame_num = preprocessing_info['frameNum']
            threshold = preprocessing_info['depthThreshold']
            rotation = preprocessing_info['rotation']
            use_edge_disntace = preprocessing_info['useEdgeDistance']
            min_edge_distance = preprocessing_info['minEdgeDistance']

            android_image = cv2.imread(os.path.join(android_image_path, f'depth_{frame_num}.png'), -1)
            android_image[android_image > threshold] = 0
            if use_edge_disntace:
                try:
                    image_mask = filter_mask(android_image, minDistEdge=min_edge_distance)
                    android_image = cv2.bitwise_and(android_image, android_image, mask=image_mask)
                except Exception as e:
                    print(e)

            android_image = standardize_projection(android_image, invert=True)
            if rotation != 0:
                android_image = imutils.rotate(android_image, rotation)
                # do it again to fix the padded square in case of rotation
                android_image = standardize_projection(android_image)

            if display_cephalic_index:
                (left_edge, max_width_position), (right_edge, max_width_position), \
                    (max_height_position, top_edge), (max_height_position, bottom_edge) = get_cephalic_index(android_image)
                cephalic_index = (left_edge - right_edge) / (top_edge - bottom_edge)
                print('android cephalic index: ', cephalic_index)

            android_image = convert_to_rgb(android_image)
            android_image = viridis(android_image)[..., :3]
            android_image = (android_image * 255).astype(np.uint8)

            if display_cephalic_index:
                android_image = cv2.circle(android_image, (left_edge, max_width_position), 2, (255, 255, 255), -1)
                android_image = cv2.circle(android_image, (right_edge, max_width_position), 2, (255, 255, 255), -1)
                android_image = cv2.circle(android_image, (max_height_position, top_edge), 2, (255, 255, 255), -1)
                android_image = cv2.circle(android_image, (max_height_position, bottom_edge), 2, (255, 255, 255), -1)

                android_image = cv2.line(android_image, (left_edge, max_width_position), (right_edge, max_width_position),
                                      (255, 255, 255), 1)
                android_image = cv2.line(android_image, (max_height_position, top_edge), (max_height_position, bottom_edge),
                                      (255, 255, 255), 1)

            cv2.imwrite('./resources/android_compare.png', android_image)
            has_android = True

    if patient_info['iPhoneID'] != '':
        iphone_parent_path = get_file_path('iphone', patient_info['iPhoneID'])
        depths_dir = [dir for dir in os.listdir(iphone_parent_path) if dir.startswith('depth')][0]
        iphone_image_path = os.path.join(iphone_parent_path, depths_dir)

        if 'preprocessing_info.json' in os.listdir(iphone_parent_path):
            with open(os.path.join(iphone_parent_path, 'preprocessing_info.json')) as f:
                preprocessing_info = json.load(f)
                print('iphone preprocessing info')
                print(preprocessing_info)

            frame_num = preprocessing_info['frameNum']
            threshold = preprocessing_info['depthThreshold']
            rotation = preprocessing_info['rotation']
            use_edge_disntace = preprocessing_info['useEdgeDistance']
            min_edge_distance = preprocessing_info['minEdgeDistance']

            iphone_image = cv2.imread(os.path.join(iphone_image_path, f'{frame_num}.exr'), -1)

            iphone_image = iphone_image[:, :, 2] * 5000
            # change type to int
            iphone_image = iphone_image.astype(np.uint16)
            iphone_image = cv2.rotate(iphone_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            iphone_image = cv2.resize(iphone_image, fx=2.5, fy=2.5, dsize=(0, 0), interpolation=cv2.INTER_NEAREST)

            iphone_image[iphone_image > threshold] = 0
            if use_edge_disntace:
                try:
                    image_mask = filter_mask(iphone_image, minDistEdge=min_edge_distance)
                    iphone_image = cv2.bitwise_and(iphone_image, iphone_image, mask=image_mask)
                except Exception as e:
                    print(e)

            iphone_image = standardize_projection(iphone_image, invert=True)
            if rotation != 0:
                iphone_image = imutils.rotate(iphone_image, rotation)
                # do it again to fix the padded square in case of rotation
                iphone_image = standardize_projection(iphone_image)

            if display_cephalic_index:
                (left_edge, max_width_position), (right_edge, max_width_position), \
                    (max_height_position, top_edge), (max_height_position, bottom_edge) = get_cephalic_index(iphone_image)
                cephalic_index = (left_edge - right_edge) / (top_edge - bottom_edge)
                print('iphone cephalic index: ', cephalic_index)

            iphone_image = convert_to_rgb(iphone_image)
            iphone_image = viridis(iphone_image)[..., :3]
            iphone_image = (iphone_image * 255).astype(np.uint8)

            if display_cephalic_index:
                iphone_image = cv2.circle(iphone_image, (left_edge, max_width_position), 2, (255, 255, 255), -1)
                iphone_image = cv2.circle(iphone_image, (right_edge, max_width_position), 2, (255, 255, 255), -1)
                iphone_image = cv2.circle(iphone_image, (max_height_position, top_edge), 2, (255, 255, 255), -1)
                iphone_image = cv2.circle(iphone_image, (max_height_position, bottom_edge), 2, (255, 255, 255), -1)

                iphone_image = cv2.line(iphone_image, (left_edge, max_width_position), (right_edge, max_width_position),
                                      (255, 255, 255), 1)
                iphone_image = cv2.line(iphone_image, (max_height_position, top_edge), (max_height_position, bottom_edge),
                                      (255, 255, 255), 1)

            cv2.imwrite('./resources/iphone_compare.png', iphone_image)
            has_iphone = True

    files = ['3dmd_compare.png' if has_3dmd else '3dmd_logo.png',
             'android_compare.png' if has_android else 'android_logo.png',
             'iphone_compare.png' if has_iphone else 'iphone_logo.png']

    # use these file names to send it to the server with a consistent name
    file_names = ['3dmd_compare.png', 'android_compare.png', 'iphone_compare.png']
    with ZipFile('./resources/comparison_files.zip', 'w') as zipObj:
        for i, file in enumerate(files):
            zipObj.write(os.path.join('./resources', file), file_names[i])

    return send_file('./resources/comparison_files.zip', mimetype='zip', as_attachment=True)


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
    prediction_mode = True
    if prediction_mode:
        from model import load_model
        model = load_model()
    app.run(debug=True)
