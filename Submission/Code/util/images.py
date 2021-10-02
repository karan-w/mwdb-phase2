'''
This class will provide all of the utility functions required for loading and saving images.
'''
class Images:

    def read_image(filepath):

    def read_images(folder_path):

    def image_arrays_to_pngs(self, images):
            index = 0
            for image in images:
                scaled_image_array = self.scale_image(image)
                image = Image.fromarray(scaled_image_array)
                image = image.convert("L")
                image_format = ".png"
                image_filename = f"image-{index}{image_format}"
                image_file_location = os.path.join(self.images_location, image_filename)
                image.save(image_file_location)
                index += 1