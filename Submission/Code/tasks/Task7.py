from zipfile import ZipFile
from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image
from scipy.spatial import distance
import matplotlib.image as img
import numpy as np
from Submission.Code.tasks.Task1 import Task1
from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
import json

if __name__=="__main__":

    base_path = Path(__file__).parent
    feature_model = "HOG"
    file_name = "image-cc-1-1.png"
    file_path = (base_path / "all/").resolve()

    with open('data_fv.json') as json_file:
        data = json.load(json_file)
    image_data = [img.imread("test_image/"+file_name)]
    fv = Task1().features(feature_model, image_data)
    qc = np.matmul((np.matmul(fv,data["U"])),data["Sigma_Inv"])

    v = np.array(data["Vt"])

    v = np.transpose(v)

    print(qc.shape)
    print(v.shape)

    print(np.matmul(qc,v))









