import numpy as np
import json

class Subject:
    def __init__(self, subject_images) -> None:
        self.images = subject_images
        
        self.subject_id = subject_images[0].subject_id

        subject_images_feature_vector = np.stack([subject_image.feature_vector for subject_image in subject_images], axis=0)
        self.feature_vector = np.mean(subject_images_feature_vector, axis=0)

        subject_images_reduced_feature_vector = np.stack([subject_image.reduced_feature_vector for subject_image in subject_images], axis=0)
        self.reduced_feature_vector = np.mean(subject_images_reduced_feature_vector, axis=0)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
