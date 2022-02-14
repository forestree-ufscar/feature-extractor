import cv2
import numpy as np


# attributes,samples,dimension,cor_RG,cor_RB,cor_RH,cor_RS,cor_RV,cor_RI,cor_GB,cor_GH,cor_GS,cor_GV,cor_GI,cor_BH,cor_BS,cor_BV,cor_BI,cor_HS,cor_HV,cor_HI,cor_SV,cor_SI,cor_VI
def extract_meta_features(img):
    # img = cv2.imread(arquivo)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = []

    B = img[:, :, 0]
    R = img[:, :, 2]
    G = img[:, :, 1]
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    data_mat = np.column_stack((
        R.reshape(-1, 1), G.reshape(-1, 1), B.reshape(-1, 1), H.reshape(-1, 1), S.reshape(-1, 1),
        V.reshape(-1, 1), gray.reshape(-1, 1)))

    # Simple

    attributes = 7  # R,G,B,H,S,V,I
    samples = gray.size
    dimension = samples / attributes
    descriptors = descriptors + [attributes, samples, dimension]

    cor_mat = np.corrcoef(data_mat, rowvar=False)  # correlation
    cor_var = []

    for i in range(0, cor_mat.shape[0]):
        for j in range(i, cor_mat.shape[1]):
            if i != j:
                cor_var = cor_var + [cor_mat[i, j]]

    descriptors = descriptors + cor_var
    return np.delete(descriptors, [6, 11, 15, 18, 21, 22])
