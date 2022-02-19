import concurrent
import os
import tempfile

import numpy as np

from feature_extractor import feature_util
from feature_extractor.feature_util import FeatureType


def extract_and_save(path, folder, max_workers=8, feature_types=None, patch_size_x=30, patch_size_y=30,
                     augmentation=False, all_patches=False, temp_folder=tempfile.gettempdir()):
    if feature_types is None:
        feature_types = [FeatureType.META, FeatureType.SASI, FeatureType.CCOM, FeatureType.COLOR, FeatureType.GIST,
                         FeatureType.GRAY, FeatureType.HOG, FeatureType.HTD, FeatureType.LAS, FeatureType.LBPRI,
                         FeatureType.QCCH, FeatureType.STEERABLE]

    print(feature_types)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print("start1")
        future = {executor.submit(export_features, path, folder, feature_type, patch_size_x, patch_size_y, augmentation,
                                  all_patches, temp_folder): feature_type for feature_type in feature_types}
        concurrent.futures.wait(future, return_when="ALL_COMPLETED")
        print("end1")
    print("end")


def export_features(path, folder, feature_type, patch_size_x, patch_size_y, augmentation, all_patches, temp_folder):
    features = feature_util.get_features_image(feature_type, path, patch_size_x, patch_size_y, augmentation, all_patches,
                                               temp_folder)
    print(os.path.join(folder, f"{feature_type}.csv"))
    np.savetxt(os.path.join(folder, f"{feature_type}.csv"), features)
