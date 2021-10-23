import numpy as np

class FeatureVector:
    def __init__(self) -> None:
        pass

    def create_dataset_feature_vector(self, images):
        images = sorted(images, key=lambda image: (image.subject_id, image.image_id)) 
        dataset_feature_vector = []
        for image in images:
            dataset_feature_vector.append(image.feature_vector)
        dataset_feature_vector = np.stack(dataset_feature_vector, axis=0) # 400 x fv_size
        return dataset_feature_vector

    def create_subjects_feature_vector(self, subjects):
        subjects = sorted(subjects, key=lambda subject: (subject.subject_id))
        subjects_feature_vector = []
        for subject in subjects:
            subjects_feature_vector.append(subject.feature_vector)
        subjects_feature_vector = np.stack(subjects_feature_vector, axis=0)
        return subjects_feature_vector

    def create_subjects_reduced_feature_vector(self, subjects):
        subjects = sorted(subjects, key=lambda subject: (subject.subject_id))
        subjects_feature_vector = []
        for subject in subjects:
            subjects_feature_vector.append(subject.reduced_feature_vector)
        subjects_feature_vector = np.stack(subjects_feature_vector, axis=0)
        return subjects_feature_vector

    '''
    This function assumes that the images are sorted by (subject_id, and image_id) 
    and the feature_vector array follows the same sorted order. It's the responsibility 
    of the caller to provide the two inputs in sorted order. 
    '''
    def assign_images_reduced_feature_vector(self, images, reduced_dataset_feature_vector):
        for index, image in enumerate(images):
            image.reduced_feature_vector = reduced_dataset_feature_vector[index]
        return images
