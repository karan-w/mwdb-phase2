import logging
import os
from os.path import isfile, join
from datetime import datetime
from types import GetSetDescriptorType
import cv2
from .image import Image
import matplotlib.image as img
import re

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs/logs.log", filemode="w", level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

'''
This class is responsible for reading images from disk.

'''
class ImageReader:
    image_filename_regex = r'image-[a-z0-9]*-\d*-\d*.png'

    def __init__(self):
        pass
    
    def sampleID(self, fileName):
        return(fileName[-5:])

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

    # def get_subject_images(self, folder_path, image_type, subject_id, number_of_images):
    #     logger.info(f"Reading images for subject {subject_id}")
    #     subject_images = []
    #     for image_id in range(1, 1 + number_of_images):
    #         image = self.get_image(folder_path, image_type, subject_id, image_id)
    #         subject_images.append(image)
    #     return subject_images

    def parse_image_filename(self, image_filename):
        tokens = image_filename.split('-')
        image_type = tokens[1]
        subject_id = int(tokens[2])
        image_id = int(tokens[3][:-4]) # Remove the .png
        return image_type, subject_id, image_id

    def get_images_by_subjects(self, folder_path, image_type):
        logger.info("Reading images for all the subjects.")
        image_filenames = self.get_all_image_filenames_for_one_type(folder_path, image_type)
        images = []
        for image_filename in image_filenames:
            image_type, subject_id, image_id = self.parse_image_filename(image_filename)
            image = self.get_image(folder_path, image_type, subject_id, image_id)
            images.append(image)

        # TODO: Sort images by subject_id and image_id
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


    def get_images_by_type(self, folder_path, subject_id):
        logger.info("Reading images for all the types.")
        image_filenames = self.get_all_image_filenames_for_one_subject(folder_path, subject_id)
        images = []

        for image_filename in image_filenames:
            image_type, subject_id, image_id = self.parse_image_filename(image_filename)
            image = self.get_image(folder_path, image_type, subject_id, image_id)
            images.append(image)

        # TODO: Sort images by subject_id and image_id

        return images 

    # def get_images_for_subjects(self, folder_path):
    #     logger.info("Reading all images for all the subjects.")
    #     image_filenames = self.get_all_image_filenames_by_subjects(folder_path)
    #     images = []

    #     for image_filename in image_filenames:
    #         image_type, subject_id, image_id = self.parse_image_filename(image_filename)
    #         image = self.get_image(folder_path, image_type, subject_id, image_id)
    #         images.append(image)

    #     # TODO: Sort images by subject_id and image_id

    #     return images 
    # def get_all_image_filenames(self, folder_path):
    #     image_filenames = [image_filename for image_filename in os.listdir(folder_path) if re.search(self.image_filename_regex, image_filename)]
    #     for image_filename in image_filenames:
    #         print(image_filename)
    #     return
    
    def get_all_images_in_folder(self, folder_path):
        logger.info("Reading all the images in the folder.")
        image_filenames = self.get_all_images_filenames_in_folder(folder_path)
        images = []

        for image_filename in image_filenames:
            image_type, subject_id, image_id = self.parse_image_filename(image_filename)
            image = self.get_image(folder_path, image_type, subject_id, image_id)
            images.append(image)

        # TODO: Sort images by subject_id and image_id

        return images 
    
    def get_all_image_filenames_for_one_type(self, folder_path, image_type):
        self.image_filename_regex_image_type = f'image-{image_type}-\d*-\d*.png'
        image_filenames = [image_filename for image_filename in os.listdir(folder_path) if re.search(self.image_filename_regex_image_type, image_filename)]
        image_filenames = sorted(image_filenames)
        return image_filenames

    def get_all_image_filenames_for_one_subject(self, folder_path, subject_id):
        self.image_filename_regex_subject_id = f'image-[a-z0-9]*-{subject_id}-\d*.png'
        image_filenames = [image_filename for image_filename in os.listdir(folder_path) if re.search(self.image_filename_regex_subject_id, image_filename)]
        image_filenames = sorted(image_filenames)
        return image_filenames

    #this is for task 3 and 4 where we need to get all images for every type and every subject respectively
    def get_all_images_filenames_in_folder(self, folder_path):
        image_filenames = [image_filename for image_filename in os.listdir(folder_path) if re.search(self.image_filename_regex, image_filename)]
        image_filenames = sorted(image_filenames)
        return image_filenames

# relation between images and  subjects

    #for task 5,6,7
    def get_query_image(self, image_filepath):
        logger.info(f"Reading image at filepath {image_filepath}")
        image_matrix = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        if image_matrix is None:
            raise Exception(f"Could not read image with the filepath {image_filepath}")
        image = Image(image_filepath, image_matrix, None, None, None, image_filepath)
        logger.debug(image.__str__())
        return image
