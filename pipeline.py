import dicom2nifti
import pydicom
import glob
import os
import pandas as pd
import json
import requests

import subprocess
import shutil

import time
from variables import *
from models import *


def get_status_csv():
    if not os.path.exists(BASE_LOG_CSV_PATH):
        df = pd.DataFrame(columns=LOG_CSV_COLUMNS)
    else:
        df = pd.read_csv(BASE_LOG_CSV_PATH)
        df = df.reset_index(drop=True)
    df = df.astype({"name": str, "predicted": str, "percentage": float, "axis_qty": int})
    return df


def save_status_csv(df):
    df.to_csv(BASE_LOG_CSV_PATH, index=False)


def check_new_patients(df, base_dir):
    if not os.path.exists(base_dir):
        print(f'Path: {base_dir} does not exist')
        return []

    df_patients = df['name'].to_list()
    ls_dir = glob.glob(f'{base_dir}/*')
    dir_patients = [ls.split('/')[-1] for ls in ls_dir]
    diff = set(dir_patients) - set(df_patients)
    list_diff = list(diff)
    
    return list_diff 


def insert_or_update_row(df, patient=None, column=None, value=None):
    if patient is None or column is None or value is None:
        return False
    
    # Search the row
    if patient in df['name'].to_list():
        df.at[df.index[df['name']==patient][0], column] = value
    # If not exists, insert the row
    else:
        row = [patient] + [value if LOG_CSV_COLUMNS[i+1] == column else None for i in range(len(LOG_CSV_COLUMNS)-1)]
        df.loc[df.index.max()+1] = row
        df = df.reset_index(drop=True)
    return df
        

def run_in_shell(command, is_python=False): 
    if is_python is True:
        process = subprocess.Popen(f'eval "$(conda shell.bash hook)" && conda activate {CONDA_ENV_NAME} && echo $CONDA_DEFAULT_ENV && {command} && conda deactivate', shell=True,  executable="/bin/bash")
    else:
        process = subprocess.Popen(f'export DISPLAY={GENERAL_DISPLAY} && {command}', shell=True,  executable="/bin/bash")
        # process = subprocess.Popen(command, shell=True,  stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
    return "OK"


def convert_dicom_to_nifti(base_input_dir=None, base_output_dir=None, patient=None, is_hmv=True):
    if patient is None:
        return "No patient parameter"
    
    if base_input_dir is not None:
        patient_dir = f'{base_input_dir}/{patient}'
    else:
        patient_dir = f'{BASE_DICOM_INPUT_DIR}/{patient}'

    if base_output_dir is None:
        base_output_dir = BASE_NII_ORIGINAL_OUTPUT_DIR

    if not os.path.exists(BASE_NII_ORIGINAL_OUTPUT_DIR):
        return (False, f'Path {base_output_dir} does not exist')
    if not os.path.exists(patient_dir):
        return (False, f'Path {patient_dir} does not exist')
    
    try:
        # Creating OUTPUT path
        output_dir = f'{base_output_dir}/{patient}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    
        ct_sub_path_name = patient
        if is_hmv is True:
            patient_dir = glob.glob(f'{patient_dir}/*')[0]
            ct_sub_path_name = patient_dir.split('/')[-1] 

        output_dir = f'{output_dir}/{ct_sub_path_name}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        study_id = []
        if is_hmv is True:
            dicom_paths = glob.glob(f'{patient_dir}/*')
            for dicom_path in dicom_paths:
                ### Getting study_id
                single_dicom_path = glob.glob(f'{dicom_path}/*')[0]
                ds = pydicom.read_file(single_dicom_path)
                study_id.append(ds.StudyInstanceUID)

                ct_name = dicom_path.split('/')[-1]
                out_file = f'{output_dir}/{ct_name}.nii.gz'
                print(f'1input: {dicom_path}\noutput: {out_file}')
                dicom2nifti.dicom_series_to_nifti(dicom_path, out_file, reorient_nifti=False)
        else:
            ### Getting study_id
            single_dicom_path = glob.glob(f'{patient_dir}/*')[0]
            ds = pydicom.read_file(single_dicom_path)
            study_id.append(ds.StudyInstanceUID)
            
            ct_name = patient_dir.split('/')[-1]
            out_file = f'{output_dir}/{ct_name}.nii.gz'
            print(f'input: {patient_dir}\noutput: {out_file}')
            dicom2nifti.dicom_series_to_nifti(patient_dir, out_file, reorient_nifti=False)


        return (True, study_id)
    except Exception as e:
        return (False, f'Error: {e}')


def phnn_segmentation(base_nii_dir=None, base_output_dir=None, patient=None, threshold=None, batch_size=None):
    input_dir = ''
    output_dir = BASE_SEGMENTED_OUTPUT_DIR 
    if base_nii_dir is not None:
        input_dir = f'{base_nii_dir}/{patient}'
    else:
        input_dir = f'{BASE_NII_ORIGINAL_OUTPUT_DIR}/{patient}'
    if base_output_dir is not None:
        output_dir = base_output_dir


    if not os.path.exists(output_dir):
        return (False, f'Path {output_dir} does not exist')
    if not os.path.exists(input_dir):
        return (False, f'Path {input_dir} does not exist')
    
    if threshold is None:
        threshold = PHNN_THRESHOLD
    if batch_size is None:
        batch_size = PHNN_BATCH_SIZE
        
    try:
        command = f'{PYTHON_PATH} {PHNN_EXECUTABLE_PATH} --directory_in {input_dir} --directory_out {output_dir} --batch_size {batch_size} --threshold {threshold}'
        print(command)
        phnn_seg = run_in_shell(command, is_python=True)
        print(phnn_seg)
        return (True, '')
    except Exception as e:
        return (False, f'Error: {e}')


def mitk_views_maker(base_nii_dir=None, base_output_dir=None, patient=None, tf_path=None, width=None, height=None, slices=None, axis=None):
    input_dir = ''
    output_dir = BASE_MITK_VIEWS_OUTPUT_DIR
    if base_nii_dir is not None:
        input_dir = f'{base_nii_dir}/{patient}'
    else:
        input_dir = f'{BASE_SEGMENTED_OUTPUT_DIR}/{patient}'
    if base_output_dir is not None:
        output_dir = base_output_dir
    if tf_path is None:
        tf_path = MITK_TRANSFER_FUNCTION_PATH
    
    if not os.path.exists(output_dir):
        return (False, f'Path {output_dir} does not exist')
    if not os.path.exists(input_dir):
        return (False, f'Path {input_dir} does not exist')
    if not os.path.exists(tf_path):
        return (False, f'Path {tf_path} does not exist')
    
    if width is None:
        width = MITK_VIEWS_WIDTH
    if height is None:
        height = MITK_VIEWS_HEIGHT 
    if slices is None:
        slices = MITK_VIEWS_LENGTH 
    if axis is None:
        axis = MITK_VIEWS_AXIS
    
    #### To get MITK_VIEWS_LENGTH/2 will be odd and MITK_VIEWS_LENGTH > 14, mandatory for script
    if slices % 2 == 1:
        slices +=1 
    if (slices / 2) % 2 == 0:
        slices += 2
    slices = max(14, slices)

    try:
        #### Creating patient views output dirs /output/patient/axisI/
        output_dir = f'{output_dir}/{patient}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for i in range(axis):
            name_axis = 'axis'+str(i+1)
            t_path = f'{output_dir}/{name_axis}'
            if not os.path.exists(t_path):
                os.mkdir(t_path)
            else:
                shutil.rmtree(t_path, ignore_errors=True)
                os.mkdir(t_path)

        #### Iterate over nifti and build command
        nii_list = glob.glob(f'{input_dir}/*/crop_by_mask*.nii.gz')
        print(f'{input_dir}/*/crop_by_mask*.nii.gz')
        print(nii_list)
        for ct_image in nii_list:
            command = f'vglrun {MITK_VIEWS_EXECUTABLE_PATH} -tf {tf_path} -i {ct_image} -o {output_dir} -w {width} -h {height} -c {slices} -a{axis}'
            # command = f'{MITK_VIEWS_EXECUTABLE_PATH} -tf {tf_path} -i {ct_image} -o {output_dir} -w {width} -h {height} -c {slices} -a{axis}'
            print(command)
            mitk_views = run_in_shell(command, is_python=False)
            print(mitk_views)
        return (True, '')
    except Exception as e:
        return (False, f'Error: {e}')


def mitk_video_maker(base_nii_dir=None, base_output_dir=None, patient=None, tf_path=None, width=None, height=None, time=None, fps=None):
    input_dir = ''
    output_dir = BASE_MITK_VIDEO_OUTPUT_DIR
    if base_nii_dir is not None:
        input_dir = f'{base_nii_dir}/{patient}'
    else:
        input_dir = f'{BASE_SEGMENTED_OUTPUT_DIR}/{patient}'
    if base_output_dir is not None:
        output_dir = base_output_dir
    if tf_path is None:
        tf_path = MITK_TRANSFER_FUNCTION_PATH
    
    if not os.path.exists(output_dir):
        return (False, f'Path {output_dir} does not exist')
    if not os.path.exists(input_dir):
        return (False, f'Path {input_dir} does not exist')
    if not os.path.exists(tf_path):
        return (False, f'Path {tf_path} does not exist')
    
    if width is None:
        width = MITK_VIDEO_WIDTH
    if height is None:
        height = MITK_VIDEO_HEIGHT 
    if time is None:
        time = MITK_VIDEO_TIME
    if fps is None:
        fps = MITK_VIDEO_FPS
    
    try:
        #### Iterate over nifti and build command
        nii_list = glob.glob(f'{input_dir}/*/crop_by_mask*.nii.gz')
        for ct_image in nii_list:
            command = f'vglrun {MITK_VIDEO_EXECUTABLE_PATH} -tf {tf_path} -i {ct_image} -o {output_dir}/{patient}.mp4 -w {width} -h {height} -t {time} -f {fps}'
            # command = f'{MITK_VIDEO_EXECUTABLE_PATH} -tf {tf_path} -i {ct_image} -o {output_dir}/{patient}.mp4 -w {width} -h {height} -t {time} -f {fps}'
            print(command)
            mitk_views = run_in_shell(command, is_python=False)
            print(mitk_views)
        return (True, f'{output_dir}/{patient}.mp4')
    except Exception as e:
        return (False, f'Error: {e}')


def process_prediction(base_model_dir=None, base_legend_dir=None, base_slices_dir=None, patient=None, width=None, height=None, axis=None, classes=None):
    if base_slices_dir is None:
        base_slices_dir = BASE_MITK_VIEWS_OUTPUT_DIR
    if base_model_dir is None:
        base_model_dir = PREDICTION_MODEL_PATH
    if base_legend_dir is None:
        base_legend_dir = PREDICTION_LEGEND_PATH
    
    if not os.path.exists(base_slices_dir):
        return (False, {'error': f'Path {base_slices_dir} does not exist'})
    if not os.path.exists(base_model_dir):
        return (False, {'error': f'Path {base_model_dir} does not exist'})
    if not os.path.exists(base_legend_dir):
        return (False, {'error': f'Path {base_legend_dir} does not exist'})
    
    if width is None:
        width = PREDICTION_WIDTH
    if height is None:
        height = PREDICTION_HEIGHT 
    if axis is None:
        axis = PREDICTION_AXIS
    if classes is None:
        classes = PREDICTION_CLASSES
    
    try:
        #### Iterate over nifti and build command
        model_name = 'resnet101'
        model_cidia = ModelCidia(base_model_dir, base_legend_dir, base_slices_dir, model_name, width, height)
        bool_status, result = model_cidia.test_patient(axis, patient)
        del model_cidia
        if not bool_status:
            return (False, result)
        return (True, result)
    except Exception as e:
        return (False, {'error': f'Error: {e}'})


def upload_video(study_id=None, video_path=None):
    if study_id is None:
        return (False, {'error': f'study_id is NULL'})
    if study_id is None:
        return (False, {'error': f'video_path is NULL'})
    
    if not os.path.exists(video_path):
        return (False, {'error': f'Path {video_path} does not exist'})
    try:
        video_name = video_path.split('/')[-1] 
        #### Iterate over study_id and build command
        for study in study_id:
            command = f'curl --request PUT --data-binary "@{video_path}" https://cidia.ufrgs.dev/storage/v1/exam-media/{study}/3d-video/{video_name}'
            video = run_in_shell(command, is_python=False)
            # print(command)
            print(video)
        return (True, '')
    except Exception as e:
        return (False, f'Error: {e}')
 
def upload_results(url=None, study_id=None, data=None):
    if url is None:
        url = API_URL
    
    if study_id is None:
        return (False, {'error': f'study_id is NULL'})
    if study_id is None:
        return (False, {'error': f'data is NULL'})
    
    try:
        #### Iterate over study_id and build command
        output = 1
        if data['percentage'] < 50.0:
            output = 2
        for study in study_id:
            payload = {'series_id': None, 'study_id': study, 'algorithm_id': 'metadata', 'output': output, 
                    'metadata': {'name':data['name'], 'radiology': '', 'volume_rendering': data['percentage']}}
            r = requests.post(url, data=json.dumps(payload))
            # print(payload)
            print(r)
        return (True, '')
    except Exception as e:
        return (False, f'Error: {e}')



def run_pipeline(patient_list=None):
    df = get_status_csv()
    # list_patients = check_new_patients(df, BASE_DICOM_INPUT_DIR)
    result = {'success': True, 'detail': ''}
    for patient in patient_list:
        print(patient)
        print('Init time: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        # CALL DICOM -> NIFTI
        a, b = None, None #dicom_path, nii_path
        if len(patient) < 5: is_hmv = True
        else: is_hmv = False
        
        study_id = None
        
        bool_result, text_out = convert_dicom_to_nifti(base_input_dir=a, base_output_dir=b, patient=patient, is_hmv=is_hmv)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='to_nifti', value=True)
            df = insert_or_update_row(df, patient=patient, column='study_id', value=text_out)
            study_id = text_out
        else:
            result['success'] = False
            result['detail'] += f'Dicom to nifti: {text_out}'
        print('Convert dicom -> nifti: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL PHNN SEGMENTATION
        c = None # nii_segmented_path
        bool_result, text_out = phnn_segmentation(base_nii_dir=b, base_output_dir=c, patient=patient, threshold=None, batch_size=None)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='segmented', value=True)
        else:
            result['success'] = False
            result['detail'] += f'\nPhnn segmentation: {text_out}'
        print('Segment nifti: ' + time.strftime("%Y-%m-%d %H:%M:%S"))

        # CALL MITK VIEWS
        d, e = None, None # slices_path, tf_file_path
        
        bool_result, text_out = mitk_views_maker(base_nii_dir=c, base_output_dir=d, patient=patient, tf_path=e, width=None, height=None, slices=None, axis=None)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='to_slices_3d', value=True)
        else:
            result['success'] = False
            result['detail'] += f'\nSlices 2d: {text_out}'
        print('slices 2d: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL MITK VIDEOS
        f = None # videos_path
        bool_result, text_out = mitk_video_maker(base_nii_dir=c, base_output_dir=f, patient=patient, tf_path=e, width=None, height=None, time=None, fps=None)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='to_video', value=True)
        else:
            result['success'] = False
            result['detail'] += f'\nVideo: {text_out}'
        print('Videos: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        bool_result, text_out = upload_video(study_id=study_id, video_path=text_out)
        if not bool_result:
            result['success'] = False
            result['detail'] += f'\nUpload video: {text_out}'
        
        # CALL PROCESS PREDICT
        g, h = None, None
        w, h, ax, cl = None, None, None, None
        bool_result, text_out = process_prediction(base_model_dir=g, base_legend_dir=h, base_slices_dir=d, patient=patient, width=w, height=h, axis=ax, classes=cl)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='predicted', value=text_out['predicted'])
            df = insert_or_update_row(df, patient=patient, column='percentage', value=text_out['percentage'])
            df = insert_or_update_row(df, patient=patient, column='axis_detail', value=text_out['axis_detail'])
            df = insert_or_update_row(df, patient=patient, column='axis_qty', value=text_out['axis_qty'])
            print(df)
            
            u = None
            text_out['name'] = patient
            bool_result, text_out = upload_results(url=u, study_id=study_id, data=text_out)
            if not bool_result:
                result['success'] = False
                result['detail'] += f'\nUpload result: {text_out}'
            else: #Success -> save dataframe
                save_status_csv(df)
        else:
            result['success'] = False
            result['detail'] += f'\nPrediction process: {text_out}'
        print('Prediction: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
    return result



###################################################
## Flask service
###################################################
#!flask/bin/python
from flask import Flask
from flask import request
import json
import sys

app = Flask(__name__)

@app.route('/pipeline3/<ct_path>', methods=['GET'])
def index(ct_path):
    if ct_path is not None:
        results = run_pipeline(patient_list=[ct_path])
        out = json.dumps(results)
    return out


# Prepare some function for massive and re-use the method
# @app.route('/pipeline3', methods=['POST'])
# def index():
#     if ct_path is not None:
#         results = run_pipeline(patient_list=[ct_path])
#         out = json.dumps(results)
#     return out




# if __name__ == '__main__':
#     app.run(debug=True)
