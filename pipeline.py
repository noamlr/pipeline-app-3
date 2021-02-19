import dicom2nifti
import pydicom
import glob
import os
import pandas as pd
import json
import requests
import pickle

import subprocess
import shutil

import time
from variables import *


def run_in_shell(command, is_python=False): 
    if is_python is True:
        process = subprocess.Popen(f'eval "$(conda shell.bash hook)" && conda activate {CONDA_ENV_NAME} && echo $CONDA_DEFAULT_ENV && {command} && conda deactivate', shell=True,  executable="/bin/bash")
    else:
        process = subprocess.Popen(f'export DISPLAY={GENERAL_DISPLAY} && {command}', shell=True,  executable="/bin/bash")
        # process = subprocess.Popen(command, shell=True,  stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
    return "OK"


def convert_dicom_to_nifti(dicom_dir=None, output_dir=None):
    try:
        if not os.path.exists(dicom_dir):
            return (False, 'Dicom dir  is NULL')

        if dicom_dir.endswith('/'):
            dicom_dir = dicom_dir[:-1]
        study_id = dicom_dir.split('/')[-2]
        serie_id = dicom_dir.split('/')[-1]
        
        if output_dir is None:
            output_dir = f'{BASE_NII_ORIGINAL_OUTPUT_DIR}/{study_id}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(output_dir):
            return (False, f'Path {output_dir} does not exist')
    
        output_dir = f'{output_dir}/{serie_id}.nii.gz'
        dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir, reorient_nifti=False)
        return (True, output_dir)
    except Exception as e:
        return (False, f'Error: {e}')


def phnn_segmentation(nii_path=None, output_dir=None, threshold=None, batch_size=None):
    try:
        if not os.path.exists(nii_path):
            return (False, f'File {nii_path} does not exist')
    
        study_id = nii_path.split('/')[-2]
        serie_id = nii_path.split('/')[-1]

        if output_dir is None:
            output_dir = f'{BASE_SEGMENTED_OUTPUT_DIR}/{study_id}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(output_dir):
            return (False, f'Path {output_dir} does not exist')
    
        if threshold is None:
            threshold = PHNN_THRESHOLD
        if batch_size is None:
            batch_size = PHNN_BATCH_SIZE
        
        command = f'{PYTHON_PATH} {PHNN_EXECUTABLE_PATH} --nii_path {nii_path} --directory_out {output_dir} --batch_size {batch_size} --threshold {threshold}'
        print(command)
        phnn_seg = run_in_shell(command, is_python=True)
        print(phnn_seg)
        return (True, f'{output_dir}/crop_by_mask_{serie_id}')
    except Exception as e:
        return (False, f'Error: {e}')


def mitk_views_maker(nii_path=None, output_dir=None, tf_path=None, width=None, height=None, slices=None, axis=None):
    try:
        if nii_path is None:
            return (False, f'Nii_path is Null')
        if not os.path.exists(nii_path):
            return (False, f'Path {nii_path} does not exist')
        
        if not nii_path.endswith('.nii.gz'):
            return (False, f'{nii_path} is not .nii.gz file')
        
        study_id = nii_path.split('/')[-2]
        serie_id = nii_path.split('/')[-1]
        serie_id = serie_id[13:-7] # remove crop_by_mask_ and .nii.gz

        if output_dir is None:
            output_dir = BASE_MITK_VIEWS_OUTPUT_DIR
        if study_id not in output_dir:
            output_dir = f'{output_dir}/{study_id}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        if serie_id not in output_dir:
            output_dir = f'{output_dir}/{serie_id}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if tf_path is None:
            tf_path = MITK_TRANSFER_FUNCTION_PATH
    
        if not os.path.exists(output_dir):
            return (False, f'Path {output_dir} does not exist')
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

        #### Creating patient views output dirs /output/axisX/
        for i in range(axis):
            name_axis = 'axis'+str(i+1)
            t_path = f'{output_dir}/{name_axis}'
            if not os.path.exists(t_path):
                os.mkdir(t_path)
            else:
                shutil.rmtree(t_path, ignore_errors=True)
                os.mkdir(t_path)

        #### Iterate over nifti and build command
        command = f'vglrun {MITK_VIEWS_EXECUTABLE_PATH} -tf {tf_path} -i {nii_path} -o {output_dir} -w {width} -h {height} -c {slices} -a{axis}'
        print(command)
        mitk_views = run_in_shell(command, is_python=False)
        print(mitk_views)
        return (True, f'{output_dir}')
    except Exception as e:
        return (False, f'Error: {e}')


def mitk_video_maker(nii_path=None, output_dir=None, tf_path=None, width=None, height=None, time=None, fps=None):
    try:
        if nii_path is None:
            return (False, f'Nii_path is Null')
        if not os.path.exists(nii_path):
            return (False, f'Path {nii_path} does not exist')
        if not nii_path.endswith('.nii.gz'):
            return (False, f'{nii_path} is not .nii.gz file')
        study_id = nii_path.split('/')[-2]
        serie_id = nii_path.split('/')[-1]
        serie_id = serie_id[13:-7] # remove crop_by_mask_ and .nii.gz

        if output_dir is None:
            output_dir = BASE_MITK_VIEWS_OUTPUT_DIR
        if study_id not in output_dir:
            output_dir = f'{output_dir}/{study_id}'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if tf_path is None:
            tf_path = MITK_TRANSFER_FUNCTION_PATH
    
        if not os.path.exists(output_dir):
            return (False, f'Path {output_dir} does not exist')
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
    
        #### Iterate over nifti and build command
        command = f'vglrun {MITK_VIDEO_EXECUTABLE_PATH} -tf {tf_path} -i {nii_path} -o {output_dir}/{serie_id}.mp4 -w {width} -h {height} -t {time} -f {fps}'
        print(command)
        mitk_views = run_in_shell(command, is_python=False)
        print(mitk_views)
        return (True, f'{output_dir}/{serie_id}.mp4')
    except Exception as e:
        return (False, f'Error: {e}')


def process_prediction(base_model_dir=None, base_legend_dir=None, base_slices_dir=None, width=None, height=None, axis=None, classes=None):
    if base_slices_dir is None:
        return (False, {'error': f'{base_slices_dir} cannot be Null'})
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
        command = f'{PYTHON_PATH} models.py --md {base_model_dir} --ld {base_legend_dir} --sd {base_slices_dir} --mn {model_name} --w {width} --h {height} --a {axis}'
        print(command)
        pred = run_in_shell(command, is_python=True)
        print(pred)
        result = None
        with open('prediction_result.pkl', 'rb') as pr:
            result = pickle.load(pr)

        if result is None or result['success'] is False:
            return (False, result)
        return (True, result)
    except Exception as e:
        return (False, {'error': f'Error: {e}'})


def run_convert_and_segment(dicom_dir=None, nii_dir=None, nii_segmented_dir=None):
    try:
        result = {'success': True, 'detail': '', 'nii_segmented_path': None}
        print('Init time: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL DICOM -> NIFTI
        study_id = None 
        bool_result, text_out = convert_dicom_to_nifti(dicom_dir=dicom_dir, output_dir=nii_dir)
        if bool_result:
            nii_dir = text_out
        else:
            result['success'] = False
            result['detail'] += f'Dicom to nifti: {text_out}'
            return result
        print('Convert dicom -> nifti: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL PHNN SEGMENTATION
        bool_result, text_out = phnn_segmentation(nii_path=nii_dir, output_dir=nii_segmented_dir, threshold=None, batch_size=None)
        if bool_result:
            nii_segmented_dir = text_out
            result['nii_segmented_path'] = text_out
        else:
            result['success'] = False
            result['detail'] += f'Phnn segmentation: {text_out}'
        print('Segment nifti: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        return result

    except Exception as e:
        return {'success': False, 'detail': f'{e}'}


def run_third_approach(nii_segmented_dir=None):
    try:
        result = {'success': True, 'detail': ''}
        print('Init time: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL MITK VIEWS
        views_dir = None
        bool_result, text_out = mitk_views_maker(nii_path=nii_segmented_dir, output_dir=None, tf_path=None, width=None, height=None, slices=None, axis=None)
        if bool_result:
            views_dir = text_out
        else:
            result['success'] = False
            result['detail'] += f'Slices 2d: {text_out}'
            return result
        print('slices 2d: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL MITK VIDEOS
        f = None # videos_path
        video_path = None
        bool_result, text_out = mitk_video_maker(nii_path=nii_segmented_dir, output_dir=None, tf_path=None, width=None, height=None, time=None, fps=None)
        if bool_result:
            video_path = text_out
            result['video_path'] = video_path
        else:
            result['success'] = False
            result['detail'] += f'Video: {text_out}'
            return result
        print('Videos: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL PROCESS PREDICT
        bool_result, text_out = process_prediction(base_model_dir=None, base_legend_dir=None, base_slices_dir=views_dir, width=None, height=None, axis=None, classes=None)
        if bool_result:
            result['predicted'] = text_out['predicted']
            result['percentage'] = text_out['percentage']
            result['axis_detail'] = text_out['axis_detail']
            result['axis_qty'] = text_out['axis_qty']
        else:
            result['success'] = False
            result['detail'] += f'\nPrediction process: {text_out}'
        print('Prediction: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        return result
    except Exception as e:
        return {'success': False, 'detail': f'{e}'}






if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Run P-HNN on nii image.')

    parser.add_argument('--dicom_dir', type=str, required=False, default=None, help='path to input dicom folder')
    parser.add_argument('--nii_out', type=str, required=False, default=None, help='path to output .nii file')
    parser.add_argument('--nii_segmented_dir', type=str, required=False, default=None, help='Segmented nifti file path')
    parser.add_argument('--convert_and_segment', required=False, default=False, help='Process only convert and segmentation', action='store_true')
    parser.add_argument('--third_approach', required=False, default=False, help='Process only convert and segmentation', action='store_true')
    parser.add_argument('--full_pipeline', required=False, default=False, help='Process only prediction', action='store_true')
    
    args = parser.parse_args()

    result = None
    segmented_path = None
    
    if args.nii_segmented_dir is not None:
        segmented_path = args.nii_segmented_dir

    if args.full_pipeline is True or args.convert_and_segment is True:
        result = run_convert_and_segment(dicom_dir=args.dicom_dir, nii_dir=args.nii_out, nii_segmented_dir=args.nii_segmented_dir)
        if result['success'] is True:
            segmented_path = result['nii_segmented_path']
        else:
            print(result)
    with open('/data/pipeline_result.pkl', 'wb') as pr:
        pickle.dump(result, pr)
    
    if args.full_pipeline is True or args.third_approach is True:
        result = run_third_approach(nii_segmented_dir=segmented_path)
        print(result)
    with open('/data/pipeline_result.pkl', 'wb') as pr:
        pickle.dump(result, pr)
