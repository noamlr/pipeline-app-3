import numpy as np
import pandas as pd
import glob
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications import ResNet101
from tensorflow import device
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras import Model 
from tensorflow.keras.models import load_model 
from tensorflow.keras.optimizers import Adam

class ModelCidia():
    def __init__(self, model_path, legend_path, slices_path, model_name, width, height):
        self.MODEL_PATH = model_path
        self.LEGEND_PATH = legend_path
        self.SLICES_PATH = slices_path
        self.MODEL_NAME = model_name
        self.WIDTH = width
        self.HEIGHT = height
        # self.status, self.MODEL = self.get_model()

    
    def get_base_model(self):
        try:
            base_model = None
            if self.MODEL_NAME == 'resnet101':
                base_model = ResNet101(weigths='imagenet', include_top=False, input_shape=(self.WIDTH, self.HEIGHT, 3))
            return (True, base_model)
        except:
            return (False, 'Error generating base model')

    def get_model(self):
        try:
            with device('/GPU:0'):
                status, base_model = self.set_base_model()
                if not status:
                    return (False, base_model)
                self.MODEL.trainable = True
                x = self.MODEL.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(0.2)(x)
                x = Dense(1024, activation='relu')(x)
                x = Dense(1024, activation='relu')(x)
                x = Dropout(0.2)(x)
                x = BatchNormalization()(x)
            
                preds = Dense(units=len(self.LEGEND), activation='softmax')(x)

                model = Model(inputs=base_model.input, outputs=preds)
                model.compile(optimizers=Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
                model.summary
                return (True, model)
        except:
            return (False, 'Error getting model')

    def test_patient(self, axis, patient):
        try:
            prediction = []
            for i range(axis):
                t_axis=i+1
                
                # Check legend path
                legend_path = f'{self.LEGEND_PATH}/axis{t_axis}/legend.npy'
                class_legend = None
                if not os.path.exists()
                    class_legend = np.load(legend_path, allow_pickle=True).item()
                    class_legend = dict((v, k) for k,v in class_legend.items())
                
                axis_name = f'axis{t_axis}'
                self.MODEL = load_model(f'{self.MODEL_PATH}/{axis_name}/my_checkpoint')
                patient_dir = f'{self.SLICES_PATH}/{patient}'
                if not os.path.exists(patient_dir):
                    return (False, f'Path: {patient_dir} does not exist')
                imgs_filename = sorted(os.listdir(patient_dir))
                test_df = pd.DataFrame({'filename': imgs_filename[:]})

                nb_samples = test_df.shape[0]

                # DataGenerator:
                test_gen = ImageDataGenerator(rescale=1./255)
                test_generator = test_gen.flow_from_dataframe(
                    test_df,
                    patient_dir,
                    x_col='filename',
                    y_col=None,
                    class_mode=None,
                    target_size=(self.WIDTH, self.HEIGTH),
                    batch_size=16,
                    shuffle=False
                )
                predict = self.MODEL.predict(test_generator, steps=np.ceil(nb_samples/16))
                test_df['predicted'] = [np.where(pr == np.max(pr))[0][0] for pr in predict]
                test_df['predicted'] = test_df['predicted'].replace(class_legend)
                test_df['count'] = 1
                test_df = test_df.group_by('predicted', as_index=False)['count'].count()
                print(test_df)

        except:
            return (False, 'Except error testing patient')
            

    def get_file_path(self, search_filter=''):
        paths = []
        for root, dirs, files in os.walk(folder):
            path = os.path.join(root, file)
            if search_filter in path:
                paths.append(path)
        return paths

    def get_data_generator(self, dataframe, x_col, y_col, subset=None, shuffle=True, batch_size=32, class_mode="binary"):
        datagen = ImageDataGenerator(
            rotation_range=15,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.05,
            horizontal_flip=False,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )

        data_generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            x_col=x_col,
            y_col=y_col,
            subset=subset,
            target_size=(self.WIDTH, self.HEIGHT),
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle
        )

        return data_generator


