import enum
import os
import struct
import tempfile

import cv2
import numpy as np
from PIL import Image
from gdal_util import image_utils

from feature_extractor import extractor_gray, extractor_color, extractor_meta

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_ccom(data, temp_folder):
    features = []
    features_size_max = 0

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'ccom_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        features_size = int.from_bytes(d[0:4], byteorder='little')
        features_size_max = max(features_size_max, features_size)

        for j in range(0, features_size):
            f.append(struct.unpack('2f', d[((j * 8) + 4):((j * 8) + 12)])[1])

        features.append(np.reshape(f, (-1, 1)))

    return features, features_size_max


def _convert_image(data, input_file):
    img = Image.fromarray(np.reshape(data, (len(data[0]), len(data[0][0]), len(data))), "RGB")
    img.save(input_file)


def _get_gabriel_gray(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        input_file = tempfile.mktemp(suffix=".ppm", dir=temp_folder)
        _convert_image(data[i], input_file)
        image = cv2.imread(input_file, 0)
        f = extractor_gray.extract_gray_features(image)
        f = np.nan_to_num(f)
        os.remove(input_file)

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_gabriel_color(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        input_file = tempfile.mktemp(suffix=".ppm", dir=temp_folder)
        _convert_image(data[i], input_file)
        image = cv2.imread(input_file)
        f = extractor_color.extract_color_features(image)
        os.remove(input_file)

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_gabriel_meta(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        input_file = tempfile.mktemp(suffix=".ppm", dir=temp_folder)
        _convert_image(data[i], input_file)
        image = cv2.imread(input_file)
        f = extractor_meta.extract_meta_features(image)
        os.remove(input_file)
        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_gist(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(np.array(data[i] / 255, dtype='uint8'), 'compute_gist', temp_folder)

        with open(temp_output_path, 'r') as output:
            d = output.readlines()[1]

        f = []
        for v in d[0:-1].split(' '):
            f.append(float(v))

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_htd(data, temp_folder):
    features = []
    features_size_max = 0

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'mpeg7_htd_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        features_size = int.from_bytes(d[0:4], byteorder='little') * int.from_bytes(d[4:8], byteorder='little')
        features_size_max = max(features_size_max, features_size)

        for j in range(0, features_size):
            f.append(struct.unpack('2f', d[((j * 8) + 8):((j * 8) + 16)])[1])

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_las(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'las_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        for v in range(0, len(d), 8):
            f.append(struct.unpack('2f', d[v:v + 8])[1])

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_sasi(data, temp_folder):
    features = []
    features_size_max = 0

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'sasi_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        features_size = int.from_bytes(d[0:4], byteorder='little')
        features_size_max = max(features_size_max, features_size)

        for j in range(0, features_size):
            f.append(struct.unpack('2f', d[((j * 8) + 4):((j * 8) + 12)])[1])

        features.append(np.reshape(f, (-1, 1)))

    return features, features_size_max


def _get_steerable(data, temp_folder):
    features = []
    features_size_max = 0

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'steerablepyramid_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        features_size = int.from_bytes(d[0:4], byteorder='little')
        features_size_max = max(features_size_max, features_size)

        for j in range(0, features_size):
            f.append(struct.unpack('2f', d[((j * 8) + 4):((j * 8) + 12)])[1])

        features.append(np.reshape(f, (-1, 1)))

    return features, features_size_max


def _get_unser(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'unser_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        for v in range(0, len(d), 8):
            f.append(struct.unpack('2f', d[v:v + 8])[1])

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_qcch(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'qcch_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        for v in range(0, len(d), 8):
            f.append(struct.unpack('2f', d[v:v + 8])[1])

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_lbpri_extraction(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'LBPri_extraction', temp_folder)

        with open(temp_output_path, 'rb') as output:
            d = output.read()

        f = []
        control = False
        for v in range(0, len(d)):
            if control:
                control = d[v] != 10

            if control:
                f.append(d[v])

            if not control:
                control = d[v] == 10

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _get_hog(data, temp_folder):
    features = []

    for i in range(0, len(data)):
        temp_output_path = _exec_feature_bin(data[i], 'hog', temp_folder)

        with open(temp_output_path, 'r') as output:
            d = output.readlines()[1]

        f = []
        for v in d[0:-1].split(' '):
            f.append(float(v))

        features.append(np.reshape(f, (-1, 1)))

    return features, len(features[0])


def _exec_feature_bin(data, feature_file, temp_folder):
    input_file = tempfile.mktemp(suffix=".ppm", dir=temp_folder)
    output_file = tempfile.mktemp(suffix=".ppm", dir=temp_folder)

    descriptor_path = os.path.join(CURRENT_DIR, "..", "descriptors_bins", feature_file)

    _convert_image(data, input_file)
    os.system(f"{descriptor_path} {input_file} {output_file}")
    os.remove(input_file)

    return output_file


def _extract_feature(patches, feature_type, temp_folder):
    if feature_type == FeatureType.CCOM:
        return _get_ccom(patches, temp_folder)
    elif feature_type == FeatureType.GRAY:
        return _get_gabriel_gray(patches, temp_folder)
    elif feature_type == FeatureType.COLOR:
        return _get_gabriel_color(patches, temp_folder)
    elif feature_type == FeatureType.META:
        return _get_gabriel_meta(patches, temp_folder)
    elif feature_type == FeatureType.GIST:
        return _get_gist(patches, temp_folder)
    elif feature_type == FeatureType.HTD:
        return _get_htd(patches, temp_folder)
    elif feature_type == FeatureType.LAS:
        return _get_las(patches, temp_folder)
    elif feature_type == FeatureType.SASI:
        return _get_sasi(patches, temp_folder)
    elif feature_type == FeatureType.STEERABLE:
        return _get_steerable(patches, temp_folder)
    elif feature_type == FeatureType.UNSER:
        return _get_unser(patches, temp_folder)
    elif feature_type == FeatureType.QCCH:
        return _get_qcch(patches, temp_folder)
    elif feature_type == FeatureType.LBPRI:
        return _get_lbpri_extraction(patches, temp_folder)
    elif feature_type == FeatureType.HOG:
        return _get_hog(patches, temp_folder)
    return None, None


def get_features_image(feature_type, path, patch_size_x=30, patch_size_y=30, augmentation=False, all_patches=False,
                       temp_folder=tempfile.gettempdir()):
    patches = image_utils.get_patches(5, patch_size_x, patch_size_y, path, augmentation=augmentation,
                                      all_patches=all_patches)

    if len(patches) <= 0:
        raise Exception("No patches found")

    features, features_size = _extract_feature(patches, feature_type, temp_folder)
    features_size_max = max(0, features_size)

    features = [np.concatenate((f, np.zeros((features_size_max - len(f), 1)))) for f in features]
    features = np.reshape(features, (len(features), len(features[0])))

    return features


class FeatureType(enum.Enum):
    CCOM = "ccom_extraction",
    GRAY = "gray",
    COLOR = "color",
    META = "meta",
    GIST = "compute_gist",
    HTD = "mpeg7_htd_extraction",
    LAS = "las_extraction",
    SASI = "sasi_extraction",
    STEERABLE = "steerablepyramid_extraction",
    UNSER = "unser_extraction",
    QCCH = "qcch_extraction",
    LBPRI = "LBPri_extraction"
    HOG = "hog"
