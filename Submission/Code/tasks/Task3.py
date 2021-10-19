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
    # k_value = int(input('Enter the value of k: '))
    # reduction_method = str(input('Choose the dimensionality reduction technique: '))

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
        # print(file)
        image_details = file.replace('.png','').split('-')
        image_type = image_details[1]
        # print(image_details)
        image_index = (int(image_details[2])-1)*10 + int(image_details[3])-1
        # image_data.append(img.imread(image))
        image_data=[img.imread("all/"+file)]
        fv = Task1.Task1().features(feature_model, image_data)
        if fv_size==0:
            fv_size = np.shape(fv.flatten())
        # print(fv)
        # print(file.split("-")[1])
        # break
        type_feature_mat[type_dict[image_type]][image_index] = fv.flatten()
        # type_feature_mat[type_dict[image_type]].append(fv.flatten())

        # if file.split("-")[1] in type_dict.keys():
        #     type_matrix[type_dict[file.split("-")[1]]].append(fv)

        # if file.split("-")[1] in type_dict.keys():
        #     type_matrix[type_dict[file.split("-")[1]]].append(fv)
    end = timer()
    print(timedelta(seconds=end-start))
    z = np.zeros(fv_size)
    for i in range(len(type_feature_mat)):
        for j in range(len(type_feature_mat[0])):
            if np.shape(type_feature_mat[i][j])==(0,):
                type_feature_mat[i][j] = [z]
    # t = np.transpose(type_feature_mat)
    
    type_feature_mat_t = np.transpose(type_feature_mat)
    

    print(len(type_feature_mat),len(type_feature_mat[0]),len(type_feature_mat_t),len(type_feature_mat_t[0]))
    
    # print(np.shape(type_feature_mat_t))
    # for i in range(len(type_feature_mat_t)):
    #     for j in range(len(type_feature_mat)):
    #         print(i,j," = ",np.shape(type_feature_mat_t[i][j]))
    # print(len(type_feature_mat),len(type_feature_mat[0]))

    # for i in range(len(type_feature_mat)):
    #     for j in range(len(type_feature_mat[i])):
    #         # print(i,j)
    #         print(i,j," = ",np.shape(type_feature_mat[i][j]))


    type_type_mat = [ [0 for x in range(len(type_dict))]for y in range(len(type_dict))]
    
    for i in range(len(type_feature_mat)):
        for j in range(len(type_feature_mat_t[0])):
            for k in range(len(type_feature_mat_t)):
                type_type_mat[i][j] += distance.cityblock(type_feature_mat[i][k], type_feature_mat_t[k][j])
    
    # type_type_mat = preprocessing.normalize(type_type_mat)
    print(type_type_mat)
    # for i in type_dict:
    #     for j in type_dict:
    #         # i_list = [item for sublist in type_feature_mat[type_dict[i]] for item in sublist]
    #         # j_list = [item for sublist in type_feature_mat[type_dict[j]] for item in sublist]
    #         # print(np.shape(i_list))
    #         # print(np.shape(j_list))
    #         # break
    #         type_type_mat[type_dict[i]][type_dict[j]] = distance.cityblock(np.concatenate(type_feature_mat[type_dict[i]]).flat,np.concatenate(type_feature_mat[type_dict[j]]).flat)
    # print(type_type_mat)
    # print(t)
    # print(np.matmul(type_feature_mat,t))

    # print(len(type_matrix))
    
    # for i in type_dict:
    #     for j in range(400):

            
    # type_type_sim=[]
    # for i in range(13):
    #     type_type_sim[i]=[]
    #     for j in range(13):
    #         dist_sum=0
    #         cnt=0
    #         for k in type_matrix[i]:
    #             for m in type_matrix[j]:
    #                 dist_sum+= distance.correlation(k.flatten(),m.flatten())
    #                 cnt+=1

    #         type_type_sim[i].append(dist_sum/cnt)

    # print(type_type_sim)



            # task1 = Task1()
    # type=0
    # while (type < 10):
    #     image_data = []
    #     y += 1
    #     for i in range(1, 11):
    #         image_label = 'image-' + image_type + '-' + str(y) + '-' + str(i) + '.png'
    #
    #         image_data.append(image.imread('' + image_label))


