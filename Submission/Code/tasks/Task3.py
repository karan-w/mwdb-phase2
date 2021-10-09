import gensim

from Submission.Code.tasks.Task1 import Task1

if __name__=="main":

    feature_model = str(input('Choose the feature model: '))

    k_value = int(input('Enter the value of k: '))
    reduction_method = str(input('Choose the dimensionality reduction technique: '))



    task1 = Task1()

    while (y < 40):
        output_dict = dict.fromkeys(['Subject', 'Weight'])
        image_data = []
        y += 1
        for i in range(1, 11):
            image_label = 'image-' + image_type + '-' + str(y) + '-' + str(i) + '.png'
            image_data.append(image.imread('' + image_label))