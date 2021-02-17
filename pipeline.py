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


def convert_and_segment(patient_list=None):
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
            # To save data
            df = insert_or_update_row(df, patient=patient, column='to_slices_3d', value=False)
            df = insert_or_update_row(df, patient=patient, column='to_video', value=False)
            df = insert_or_update_row(df, patient=patient, column='predicted', value='')
            df = insert_or_update_row(df, patient=patient, column='percentage', value=0.0)
            df = insert_or_update_row(df, patient=patient, column='axis_detail', value=[])
            df = insert_or_update_row(df, patient=patient, column='axis_qty', value=0)
            save_status_csv(df)
        else:
            result['success'] = False
            result['detail'] += f'\nPhnn segmentation: {text_out}'
        print('Segment nifti: ' + time.strftime("%Y-%m-%d %H:%M:%S"))

    return result

if __name__ == '__main__':
    import argparse
    import os
    
    try:
        parser = argparse.ArgumentParser(description='Convert to nii and segment with P-HNN.')

        parser.add_argument('--patient', type=str, required=True, help='patient name, sample --patient TYP-028')

        args = parser.parse_args()
        if args.patient is not None:
            results = convert_and_segment(patient_list=[args.patient])
        else:
            print("Patient could not be NULL")
    except Exception as e:
         print(f'Error: {e}')



#     app.run(debug=True)
