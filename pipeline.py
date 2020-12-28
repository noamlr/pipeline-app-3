import dicom2nifti
import glob
import os
import pandas as pd

import subprocess
import shutil

#import time
from variables import *


def get_status_csv():
    if not os.path.exists(BASE_LOG_CSV_PATH):
        df = pd.DataFrame(columns=LOG_CSV_COLUMNS)
    else:
        df = pd.read_csv(BASE_LOG_CSV_PATH)
        df = df.reset_index(drop=True)
    return df


def save_status_csv(df):
    df.to_csv(BASE_LOG_CSV_PATH, index=False)


def check_new_patients(df, base_dir):
    if not os.path.exists(base_dir):
        print(f'Path: {base_dir} does not exist')
        return []

    df_patients = df['name'].to_list()
    ls_dir = glob.glob(f'{base_dir}/*/')
    dir_patients = [ls.split('/')[-1] for ls in ls_dir]
    
    diff = set(dir_patients) - set(df_patients)
    list_diff = list(diff)
    
    return list_diff 


def insert_or_update_row(df, patient=None, column=None, value=None):
    if patient is None or column is None or value is None:
        return False

    # Search the row
    if patient in df['patient']:
        df.loc[df.index[df['patient']==patient], column] = value
    # If not exists, insert the row
    else:
        row = [patient] + [value if LOG_CSV_COLUMNS[i+1] == column else None for i in range(len(LOG_CSV_COLUMNS)-1)]
        df.loc[df.index.max()+1] = row
        df = df.reset_index(drop=True)
    return df
        

def run_in_shell(command): 
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
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

    if base_output_dir is not None:
        BASE_NII_ORIGINAL_DIR = base_output_dir

    if not os.path.exists(BASE_NII_ORIGINAL_DIR):
        return (False, f'Path {BASE_NII_ORIGINAL_DIR} does not exist')
    if not os.path.exists(patient_dir):
        return (False, f'Path {patient_dir} does not exist')
    
    try:
        # Creating OUTPUT path
        output_dir = f'{BASE_NII_ORIGINAL_DIR}/{patient}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    
        ct_sub_path_name = patient
        if is_hmv is True:
            patient_dir = glob.glob(f'{patient_dir}/*')[0]
            ct_sub_path_name = patient_dir.split('/')[-1] 

        output_dir = f'{output_dir}/{ct_sub_path_name}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if is_hmv is True:
            dicom_paths = glob.glob(f'{patient_dir}/*')
            for dicom_path in dicom_paths:
                ct_name = dicom_path.split('/')[-1]
                out_file = f'{output_dir}/{ct_name}.nii.gz'
                print(f'input: {dicom_path}\noutput: {out_file}')
                dicom2nifti.dicom_series_to_nifti(dicom_path, out_file, reorient_nifti=False)
        else:
            ct_name = patient_dir.split('/')[-1]
            out_file = f'{output_dir}/{ct_name}.nii.gz'
            print(f'input: {patient_dir}\noutput: {out_file}')
            dicom2nifti.dicom_series_to_nifti(patient_dir, out_file, reorient_nifti=False)


        return (True, '')
    except:
        return (False, 'Error')


def phnn_segmentation(base_nii_dir=None, base_output_dir=None, patient=None, threshold=None, batch_size=None):
    input_dir = ''
    output_dir = BASE_SEGMENTED_OUTPUT_DIR 
    if base_nii_dir is not None:
        input_dir = f'{base_nii_dir}/{patient}'
    else:
        input_dir = f'{BASE_DICOM_INPUT_DIR}/{patient}'
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
        command = f'{BASE_NII_ORIGINAL_OUPUT_DIR} --directory_in {input_dir} --directory_out {output_dir} --batch_size {batch_size} --threshold {threshold}'
        print(command)
        # phnn_seg = run_in_shell(command)
        # print phnn_seg
        return (True, '')
    except:
        return (False, 'Error')


def mitk_views_maker(base_nii_dir=None, base_output_dir=None, patient=None, tf_path=None, width=None, height=None, slices=None, axis=None):
    input_dir = ''
    output_dir = BASE_MITK_VIEWS_OUTPUT_DIR
    if base_nii_dir is not None:
        input_dir = f'{base_nii_dir}/{patient}'
    else:
        input_dir = f'{BASE_DICOM_INPUT_DIR}/{patient}'
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
        for ct_image in nii_list:
            command = f'{MITK_VIEWS_EXECUTABLE_PATH} -tf {tf_path} -i {ct_image} -o {output_dir} -w {width} -h {height} -c {slices} -a{axis}'
            print(command)
            mitk_views = run_in_shell(command)
            print(mitk_views)
        return (True, '')
    except:
        return (False, 'Error')


def mitk_video_maker(base_nii_dir=None, base_output_dir=None, patient=None, tf_path=None, width=None, height=None, time=None, fps=None):
    input_dir = ''
    output_dir = BASE_MITK_VIEWS_OUTPUT_DIR
    if base_nii_dir is not None:
        input_dir = f'{base_nii_dir}/{patient}'
    else:
        input_dir = f'{BASE_DICOM_INPUT_DIR}/{patient}'
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
            command = f'{MITK_VIDEO_EXECUTABLE_PATH} -tf {tf_path} -i {ct_image} -o {output_dir}/{patient}.mp4 -w {width} -h {height} -t {time} -f {fps}'
            print(command)
            mitk_views = run_in_shell(command)
            print(mitk_views)
        return (True, '')
    except:
        return (False, 'Error')


def process_prediction(base_model_dir=None, base_legend_dir=None, base_slices_dir=None, base_model_dir=None, patient=None, width=None, height=None, axis=None):
    input_dir = ''
    if base_slices_dir is not None:
        input_dir = f'{base_slices_dir}/{patient}'
    else:
        input_dir = f'{BASE_MITK_VIEWS_OUTPUT_DIR}/{patient}'
    if base_model_dir is None:
        base_model_dir = PREDICTION_MODEL_PATH
    if base_legend_dir is None:
        base_legend_dir = PREDICTION_LEGEND_PATH
    
    if not os.path.exists(input_dir):
        return (False, f'Path {input_dir} does not exist')
    if not os.path.exists(base_model_dir):
        return (False, f'Path {base_model_dir} does not exist')
    if not os.path.exists(base_legend_dir):
        return (False, f'Path {base_legend_dir} does not exist')
    
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
        if not model:
            model = ModelCidia(input_model_path, input_legend_path, input_slices_path, model, width, height)
        result = model.test(axis, patient)
        if result['error'] is not None:
            return (False, result)
        return (True, result)
    except:
        return (False, {'error': 'Try/Except error'})


if __name__ == "__main__":
    df = get_status_csv()
    list_patients = check_new_patients(df, BASE_DICOM_INPUT_DIR)

    for patient in list_patients:
        # CALL DICOM -> NIFTI
        a = '/home/noa/Documents/UFRGS/PROJECTS/CIDIA19/test-script/data/dicom-original/exame-pulmao'
        b = '/home/noa/Documents/UFRGS/PROJECTS/CIDIA19/test-script/data/nii-original/exame-pulmao'
        bool_result, text_out = convert_dicom_to_nifti(base_input_dir=a, base_output_dir=b, patient=patient, is_hmv=True)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='to_nifti', value=True)
        else:
            print(text_out)
    
        # CALL PHNN SEGMENTATION
        c = '/home/noa/Documents/UFRGS/PROJECTS/CIDIA19/test-script/data/nii-segmented/exame-pulmao'
        bool_result, text_out = phnn_segmentation(base_nii_dir=b, base_output_dir=c, patient=patient, threshold=None, batch_size=None)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='segmented', value=True)
        else:
            print(text_out)

        # CALL MITK VIEWS
        d = '/home/noa/Documents/UFRGS/PROJECTS/CIDIA19/test-script/data/slices2d/exame-pulmao'
        e = '/home/noa/Documents/UFRGS/PROJECTS/CIDIA19/test-script/data/tf/tf12_2.xml'
        bool_result, text_out = mitk_views_maker(base_nii_dir=c, base_output_dir=d, patient=patient, tf_path=e, width=None, height=None, slices=None, axis=None)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='to_slices_3d', value=True)
        else:
            print(text_out)

        # CALL MITK VIDEOS
        f = '/home/noa/Documents/UFRGS/PROJECTS/CIDIA19/test-script/data/videos'
        bool_result, text_out = mitk_video_maker(base_nii_dir=c, base_output_dir=f, patient=patient, tf_path=e, width=None, height=None, time=None, fps=None)
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='to_video', value=True)
        else:
            print(text_out)

        # CALL PROCESS PREDICT
        if bool_result:
            df = insert_or_update_row(df, patient=patient, column='predicted', value=text_out['predicted'])
            df = insert_or_update_row(df, patient=patient, column='percentage', value=text_out['percentage'])
            df = insert_or_update_row(df, patient=patient, column='axis_detail', value=text_out['axis_detail'])
            df = insert_or_update_row(df, patient=patient, column='axis_qty', value=text_out['axis_qty'])
        else:
            print(text_out['error'])



    


