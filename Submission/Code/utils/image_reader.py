import logging
import os
from os.path import isfile, join
from datetime import datetime
from types import GetSetDescriptorType
import cv2
from .image import Image
import matplotlib.image as img

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

'''
This class is responsible for reading images from disk.

'''
class ImageReader:
    def __init__(self):
        pass
    
    def get_image(self, folder_path, image_type, subject_id, image_id):
        image_filename = f'image-{image_type}-{subject_id}-{image_id}.png'
        image_filepath = os.path.join(folder_path, image_filename)
        logger.info(f"Reading image at filepath {image_filepath}")
        image_matrix = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        if image_matrix is None:
            raise Exception(f"Could not read image with the filepath {image_filepath}")
        image = Image(image_filename, image_matrix, subject_id, image_id, image_type, image_filepath)
        logger.debug(image.__str__())
        return image

    def get_subject_images(self, folder_path, image_type, subject_id):
        logger.info(f"Reading images for subject {subject_id}")
        subject_images = []
        # for image_id in range(1, 11):
        for image_id in range(1, 10):
            image = self.get_image(folder_path, image_type, subject_id, image_id)
            subject_images.append(image)
        return subject_images

    def get_images(self, folder_path, image_type):
        logger.info("Reading images for all the subjects.")
        images = []
        for subject_id in range(1, 41):
            subject_images = self.get_subject_images(folder_path, image_type, subject_id)
            for subject_image in subject_images:
                images.append(subject_image)
        return images

    def get_images2(self, folder_path):
        logger.info("Reading all images from the folder.")

        files = [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]

        images = []
        for file in files:
            image_details = file.replace('.png', '').split('-')
            image_type = image_details[1]
            image_index = (int(image_details[2]) - 1) * 10 + int(image_details[3]) - 1

            image_subject = image_details[2]
            image_data = [img.imread(folder_path+"/"+file)]
            images.append(image_data)
            image = self.get_image(folder_path,image_type,image_subject,image_details[3])
            images.append(image)
        return images