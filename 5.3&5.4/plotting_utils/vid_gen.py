import glob
import os
from cv2 import cv2

def gen_video():
        """
        Generates video, if the gen_images and additional_images functions were used. Stores under project.avi name.
        """
        pathIn = 'images'
        # files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f))]
        data_path = os.path.join(pathIn, '*' + '.png')
        jpg_files = glob.glob(data_path)
        # for sorting the file names properly
        # files.sort(key=lambda x: int(x[5:-4]))
        # jpg_files.sort()
        filename_float_mapping = {}

        for filename in jpg_files:
            filename_float_mapping[filename] = float(filename.split('\\')[-1].split('.png')[0])
        
        sorted_filenames = sorted(filename_float_mapping, key=filename_float_mapping.get)

        filename = jpg_files[0]
        img = cv2.imread(filename)
        height, width, _ = img.shape
        size = (width, height)

        out = cv2.VideoWriter(pathIn+'/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, size)

        extra_frames = 0
        for i in range(extra_frames):
            filename = sorted_filenames[0]
            # filename = os.path.join(pathIn, filename)
            # reading each files
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width, height)
            # writing to a image array
            out.write(img)

        for i in range(len(sorted_filenames)):
            filename = sorted_filenames[i]
            # filename = os.path.join(pathIn, filename)
            # reading each files
            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width, height)
            # writing to a image array
            out.write(img)
        
        extra_frames = 0
        for i in range(extra_frames):
            out.write(img)

        out.release()