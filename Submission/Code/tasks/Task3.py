from zipfile import ZipFile
from os import listdir
from os.path import isfile, join
from pathlib import Path
from PIL import Image
from scipy.spatial import distance
import matplotlib.image as img

from Submission.Code.tasks.Task1 import Task1

if __name__=="__main__":
    print("ffffffffffffffffffffffffffffffffffff")
    type_dict={"cc":0, "con":1, "detail":2, "emboss":3, "jitter":4, "neg":5, "noise1":6, "noise2":7, "original":8, "poster":9, "rot":10, "smooth":11, "stipple":12}
    
    feature_model = str(input('Choose the feature model: '))
    k_value = int(input('Enter the value of k: '))
    reduction_method = str(input('Choose the dimensionality reduction technique: '))

    base_path = Path(__file__).parent

    file_name = "phase2_data.zip"
    # zip_file_path = (base_path / "../../../phase2_data.zip").resolve()
    with ZipFile(file_name, 'r') as ziip:
        ziip.extractall()

    file_path = (base_path / "all/").resolve()
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    image_data=[]

    type_matrix=[]
    for i in range(13):
        type_matrix.append([])

    for file in files:
        print(file)
        image = Image.open("all/"+file)
        print(image)

        # image_data.append(img.imread(image))
        image_data.append(img.imread("all/"+file))
        fv = Task1().features(feature_model, image_data)
        print(file.split("-")[1])

        # if file.split("-")[1] in type_dict.keys():
        #     type_matrix[type_dict[file.split("-")[1]]].append(fv)

        if file.split("-")[1] in type_dict.keys():
            type_matrix[type_dict[file.split("-")[1]]].append(fv)

    type_type_sim=[]
    for i in range(13):
        type_type_sim[i]=[]
        for j in range(13):
            dist_sum=0
            cnt=0
            for k in type_matrix[i]:
                for m in type_matrix[j]:
                    dist_sum+= distance.correlation(k.flatten(),m.flatten())
                    cnt+=1

            type_type_sim[i].append(dist_sum/cnt)

    print(type_type_sim)



            # task1 = Task1()
    # type=0
    # while (type < 10):
    #     image_data = []
    #     y += 1
    #     for i in range(1, 11):
    #         image_label = 'image-' + image_type + '-' + str(y) + '-' + str(i) + '.png'
    #
    #         image_data.append(image.imread('' + image_label))


