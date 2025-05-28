# Data transformation module for the data processing pipeline.

import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import setup_logger, get_logger
from src.utils import save_object
setup_logger()
logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Function performs the data transformation.
        """
        try:
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = ['gender', 'race/ethnicity',
                                   'parental level of education', 'lunch', 'test preparation course']
            num_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder())
                ]

            )

            logger.info(f'Numerical columns are: {numerical_columns}')
            logger.info(f'Categorical columns are: {categorical_columns}')
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)

                ]
            )
            logger.info("Numerical Columns scaling completed")
            logger.info("Categorical Columns Encoding completed")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("The read train and test data completed.")

            logger.info('Obtaining Preprocessing object.')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'math score'
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = ['gender', 'race/ethnicity',
                                   'parental level of education', 'lunch', 'test preparation course']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying the preprocessing obejct on train and test dataframe")
            import pdb
            pdb.set_trace()
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_array, np.array(target_feature_test_df)
            ]

            logger.info("Saved preprocessing obj")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e, sys)
