from zipfile import ZipFile
from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image
from scipy.spatial import distance
import matplotlib.image as img
import numpy as np
import Task1
from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing



if __name__=="__main__":
    type_dict={"cc":0, "con":1, "emboss":2, "jitter":3, "neg":4, "noise01":5, "noise02":6, "original":7, "poster":8, "rot":9, "smooth":10, "stipple":11}

    feature_model = str(input('Choose the feature model: '))
    k_value = int(input('Enter the value of k: '))
    reduction_method = str(input('Choose the dimensionality reduction technique: '))

    base_path = Path(__file__).parent

    file_name = "phase2_data.zip"
    # zip_file_path = (base_path / "../../../phase2_data.zip").resolve()
    with ZipFile(file_name, 'r') as ziip:
        ziip.extractall()

    file_path = (base_path / "../../../all/").resolve()
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    image_data=[]

    type_matrix=[]
    type_feature_mat = [[[] for i in range(400)] for j in range(12)]
    # type_feature_mat = [[] for i in range(12)]

    for i in range(13):
        type_matrix.append([])
        

    start = timer()
    fv_size = 0
    for file in files:
        image_details = file.replace('.png','').split('-')
        image_type = image_details[1]
        image_index = (int(image_details[2])-1)*10 + int(image_details[3])-1
        image_data=[img.imread("all/"+file)]
        fv = Task1.Task1().features(feature_model, image_data)
        if fv_size==0:
            fv_size = np.shape(fv.flatten())
        type_feature_mat[type_dict[image_type]][image_index] = fv.flatten()
    
    z = np.zeros(fv_size)
    for i in range(len(type_feature_mat)):
        for j in range(len(type_feature_mat[0])):
            if np.shape(type_feature_mat[i][j])==(0,):
                type_feature_mat[i][j] = [z]
    
    type_feature_mat_t = np.transpose(type_feature_mat)
    


    type_type_mat = [ [0 for x in range(len(type_dict))]for y in range(len(type_dict))]
    type_type_sim = type_type_mat
    max_val = 0
    for i in range(len(type_feature_mat)):
        for j in range(len(type_feature_mat_t[0])):
            for k in range(len(type_feature_mat_t)):
                type_type_mat[i][j] += distance.cityblock(type_feature_mat[i][k], type_feature_mat_t[k][j])
    max_value = max(max(type_type_mat, key=max))
    
    for i in range(len(type_dict)):
        for j in range(len(type_dict)):
            type_type_sim[i][j] = 1 - (type_type_mat[i][j]/max_value)
    # print(type_type_sim)

    dr = Task1.Task1().dimension_red(reduction_method,type_type_sim,k_value)
    print(dr)
    end = timer()
    print(timedelta(seconds=end-start))

