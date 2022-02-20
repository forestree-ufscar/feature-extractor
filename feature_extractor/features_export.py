import concurrent
import logging
import os
import tempfile
import traceback

import numpy as np
import pandas as pd

from feature_extractor import feature_util
from feature_extractor.feature_util import FeatureType


def extract_and_save(path, folder, max_workers=8, feature_types=None, patch_size_x=30, patch_size_y=30,
                     augmentation=False, all_patches=False, temp_folder=tempfile.gettempdir()):
    if feature_types is None:
        feature_types = [FeatureType.META, FeatureType.SASI, FeatureType.CCOM, FeatureType.COLOR, FeatureType.GIST,
                         FeatureType.GRAY, FeatureType.HOG, FeatureType.HTD, FeatureType.LAS, FeatureType.LBPRI,
                         FeatureType.QCCH, FeatureType.STEERABLE]

    logging.info(f"feature_types: {feature_types}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future = {executor.submit(export_features, path, folder, feature_type, patch_size_x, patch_size_y, augmentation,
                                  all_patches, temp_folder): feature_type for feature_type in feature_types}
        concurrent.futures.wait(future, return_when="ALL_COMPLETED")
    logging.info(f"all features were exported to: {folder}")


def export_features(path, folder, feature_type, patch_size_x, patch_size_y, augmentation, all_patches, temp_folder):
    try:
        logging.info(f"feature_type: {feature_type}")
        features = feature_util.get_features_image(feature_type, path, patch_size_x, patch_size_y, augmentation,
                                                   all_patches, temp_folder)
        dst = os.path.join(folder, f"{feature_type}.csv")
        logging.info(f"feature_type {feature_type} export to: {dst}")
        df = pd.DataFrame(features)
        df.to_csv(dst)
    except Exception as e:
        logging.error(f"error: {e}")
        logging.error(f"traceback: {traceback.print_exc()}")
