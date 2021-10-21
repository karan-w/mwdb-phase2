import numpy as np

class Subject:
    def __init__(self, subject_images) -> None:
        self.images = subject_images
        self.subject_id = subject_images[0].subject_id
        subject_images_feature_vector = np.stack([subject_image.feature_vector for subject_image in subject_images], axis=0)
        self.feature_vector = np.mean(subject_images_feature_vector, axis=0)
