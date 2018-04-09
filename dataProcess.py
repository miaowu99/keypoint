# -*- coding: utf-8 -*-

"""

"""
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
from PIL import Image, ImageEnhance, ImageFilter


class DataGenerator():
    """
    To process images and labels
    """
    def __init__(self, dress_type, joints_list, img_dir, train_data_file='train/Annotations/train.csv', test_data_file='test/test.csv'):
        """Initializer
            Args:
            dress_tpye          : Tpye of dress
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data

        """
        self.joints_list = joints_list
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.dress_type = dress_type
        self.test_table = []  # The names of images being tested
        self.train_table = []  # The names of images being trained
        self.data_dict = {}  # The labels of images
        self.letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

    # --------------------Generator Initialization Methods ---------------------

    def creator(self):
        self._read_train_data()
        self._randomize()
        self._create_sets()

    def _read_train_data(self):
        """
        To read labels in csv
        """
        label_file = pd.read_csv(self.train_data_file)
        print('READING LABELS OF TRAIN DATA')
        for i in range(label_file.shape[0]):
            if label_file.at[i, 'image_category'] == self.dress_type:  # Only take the type we want
                joints = []
                name = str(label_file.at[i, 'image_id'])
                weight = []
                box = []
                for joint_name in self.joints_list:
                    joint_value = []
                    value = str(label_file.at[i, joint_name])
                    value = value.split('_')
                    # print(value)
                    joint_value.append(int(value[0]))
                    joint_value.append(int(value[1]))
                    joints.append(joint_value)
                    if value[2] == '-1':
                        weight.append(0)
                    else:
                        weight.append(1)
                # box of body,[x_box_min,y_box_min,x_box_max,y_box_max]
                box.append(self._min_point(joints, 0))
                box.append(self._min_point(joints, 1))
                box.append(max([x[0] for x in joints]))
                box.append(max([x[1] for x in joints]))
                # print(box)
                # print(name)
                joints = np.reshape(joints, (-1, 2))
                self.data_dict[name] = {'box': box, 'joints': joints, 'weights': weight}
                self.train_table.append(name)
        print('FINISH')
        return [self.train_table, self.data_dict]

    def read_test_data(self):
        label_file = pd.read_csv(self.test_data_file)
        print('READING LABELS OF TEST DATA')
        for i in range(label_file.shape[0]):
            if label_file.at[i, 'image_category'] == self.dress_type:  # Only take the type we want
                name = str(label_file.at[i, 'image_id'])
                self.test_table.append(name)
        print('FINISH')
        return self.test_table


    """
    Get the least number of a column of the dataFrame 
    由于现在存在不在图中的关键点，所以box的确定方面还有点问题，直接用能看到的关键点确定box会不会使训练的模型准确度降低？
    """
    def _min_point(self, joints, n):
        min_point = 600
        for joint in joints:
            temp_point = joint[n]
            if 0 < temp_point < min_point:
                min_point = temp_point
        return min_point

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def _complete_sample(self, name):
        """ Check if a sample has no missing value
        Args:
            name 	: Name of the sample
        """
        for i in range(self.data_dict[name]['joints'].shape[0]):
            if np.array_equal(self.data_dict[name]['joints'][i], [-1, -1]):
                return False
        return True

    def _give_batch_name(self, batch_size=16, set='train'):
        """ Returns a List of Samples
        Args:
            batch_size	: Number of sample wanted
            set			: Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/valid')
                break
        return list_file

    def _create_sets(self, validation_rate=0.1):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """
        sample = len(self.train_table)
        valid_sample = int(sample * validation_rate)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = []
        preset = self.train_table[sample - valid_sample:]
        print('START SET CREATION')
        for elem in preset:
            if self._complete_sample(elem):
                self.valid_set.append(elem)
            else:
                self.train_set.append(elem)
        print('SET CREATED')
        np.save('Dataset-Validation-Set', self.valid_set)
        np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

    def generate_set(self, rand=False, validationRate=0.1):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._read_train_data()
        if rand:
            self._randomize()
        self._create_sets(validation_rate=validationRate)

    # ---------------------------- Generating Methods --------------------------

    def _make_gaussian(self, height, width, sigma=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]  # 把一行数变成一列数
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_hm(self, height, width, joints, maxlength, weight):
        """ Generate a full Heap Map for every joints in an array
        Args:
            height			: Wanted Height for the Heat Map
            width			: Wanted Width for the Heat Map
            joints			: Array of Joints
            maxlength		: Length of the Bounding Box
        """
        num_joints = joints.shape[0]
        hm = np.zeros((height, width, num_joints), dtype=np.float32)
        for i in range(num_joints):
            if not (np.array_equal(joints[i], [-1, -1])) and weight[i] == 1:
                s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2
                hm[:, :, i] = self._make_gaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def _crop_data(self, height, width, box, boxp=0.05):
        """ Automatically returns a padding vector and a bounding box given
        the size of the image and a list of joints.
        Args:
            height		: Original Height
            width		: Original Width
            box			: Bounding Box
            joints		: Array of joints
            boxp		: Box percentage (Use 20% to get a good bounding box)
        """
        padding = [[0, 0], [0, 0], [0, 0]]
        # 把裁剪窗口按照人的box向外扩展20%
        crop_box = [box[0] - int(boxp * (box[2] - box[0])), box[1] - int(boxp * (box[3] - box[1])),
                    box[2] + int(boxp * (box[2] - box[0])), box[3] + int(boxp * (box[3] - box[1]))]
        if crop_box[0] < 0: crop_box[0] = 0
        if crop_box[1] < 0: crop_box[1] = 0
        if crop_box[2] > width - 1: crop_box[2] = width - 1
        if crop_box[3] > height - 1: crop_box[3] = height - 1
        new_h = int(crop_box[3] - crop_box[1])
        new_w = int(crop_box[2] - crop_box[0])
        crop_box = [crop_box[0] + new_w // 2, crop_box[1] + new_h // 2, new_w, new_h]
        if new_h > new_w:
            bounds = (crop_box[0] - new_h // 2, crop_box[0] + new_h // 2)
            if bounds[0] < 0:
                padding[1][0] = abs(bounds[0])
            if bounds[1] > width - 1:
                padding[1][1] = abs(width - bounds[1])
        elif new_h < new_w:
            bounds = (crop_box[1] - new_w // 2, crop_box[1] + new_w // 2)
            if bounds[0] < 0:
                padding[0][0] = abs(bounds[0])
            if bounds[1] > height - 1:
                padding[0][1] = abs(height - bounds[1])
        crop_box[0] += padding[1][0]
        crop_box[1] += padding[0][0]
        # padding[0]为高度（y）方向的padding，padding[1]为宽度（x）方向的padding
        return padding, crop_box

    def _crop_data_new(self, height, width):
        """ Automatically returns a padding vector
            Args:
                height		: Original Height
                width		: Original Width
                crop_box    : the center point and final point of crop box
        """
        padding = [[0, 0], [0, 0], [0, 0]]
        crop_box = [width//2, width//2, width-2, width-2]
        if width == height:
            pass
        elif width > height:
            pad_size = (width - height) // 2
            padding[0][0] = padding[0][1] = pad_size
        else:
            pad_size = (height - width) // 2
            padding[1][0] = padding[1][1] = pad_size
            crop_box[2] = crop_box[3] = height-2
            crop_box[1] = crop_box[0] = height//2
        return padding, crop_box


    def _crop_img(self, img, padding, crop_box):
        """ Given a bounding box and padding values return cropped image
        Args:
            img			: Source Image
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        max_length = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_length // 2:crop_box[1] + max_length // 2,
                crop_box[0] - max_length // 2:crop_box[0] + max_length // 2]
        return img

    def _crop(self, img, hm, padding, crop_box):
        """ Given a bounding box and padding values return cropped image and heatmap
        Args:
            img			: Source Image
            hm			: Source Heat Map
            padding	: Padding
            crop_box	: Bounding Box
        """
        img = np.pad(img, padding, mode='constant')
        hm = np.pad(hm, padding, mode='constant')
        max_length = max(crop_box[2], crop_box[3])
        img = img[crop_box[1] - max_length // 2:crop_box[1] + max_length // 2,
              crop_box[0] - max_length // 2:crop_box[0] + max_length // 2]
        hm = hm[crop_box[1] - max_length // 2:crop_box[1] + max_length // 2,
             crop_box[0] - max_length // 2:crop_box[0] + max_length // 2]
        return img, hm

    def _relative_joints(self, box, padding, joints, to_size=64):
        """ Convert Absolute joint coordinates to crop box relative joint coordinates
        (Used to compute Heat Maps)
        Args:
        box			: Bounding Box
            padding	: Padding Added to the original Image
            to_size	: Heat Map wanted Size
        """
        new_j = np.copy(joints)
        max_l = max(box[2], box[3])
        new_j = new_j + [padding[1][0], padding[0][0]]
        new_j = new_j - [box[0] - max_l // 2, box[1] - max_l // 2]
        new_j = new_j * to_size / (max_l + 0.0000001)
        return new_j.astype(np.int32)

    # ---------------------- Augmentation -----------------------------------------

    def _rotate_augment(self, img, hm, max_rotation=30):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            img = transform.rotate(img, r_angle, preserve_range=True)
            hm = transform.rotate(hm, r_angle)
        return img, hm

    def _random_erase(self, img, list_of_key, weight_of_key, probability=0.5, max_size=0.3):
        """
        Randomly erase a rectangle region of a picture and let weight of keypoint in the region be zero
        :param img:
        :param list_of_key:
        :param weight_of_key:
        :param probability:
        :param max_size:
        :return: output image, new_weight_of_key
        """
        if np.random.random() > probability:
            return img, weight_of_key
        img_width = img.shape[1]
        img_height = img.shape[0]
        img_channel = img.shape[2]
        rec_width = int(img_width*max_size*np.random.random())
        rec_height = int(img_height*max_size*np.random.random())
        rectangle = (255 * np.random.random((rec_height, rec_width, img_channel))).astype(np.int8)
        left_up = [0, 0]
        left_up[0] = int((img_width-rec_width-1) * np.random.random())
        left_up[1] = int((img_height-rec_height-1) * np.random.random())
        for i in range(len(list_of_key)):
            if left_up[0] < list_of_key[i][0] < left_up[0]+rec_width and left_up[1] < list_of_key[i][1] < left_up[1]+rec_height:
                weight_of_key[i] = 0
        img[left_up[1]:left_up[1]+rec_height, left_up[0]:left_up[0]+rec_width] = rectangle
        print('erase_rectangle:', left_up, '  width:', rec_width, '  height:', rec_height)
        return img, weight_of_key

    def _color_augment(self, img, max_brightness_rate=2.0, max_color_rate=2.0, max_contrast_rate=2.0,
                       max_sharpness_rate=3.0):
        image = Image.fromarray(img)
        # image.show()
        # 亮度增强
        if random.choice([0, 1]):
            enh_bri = ImageEnhance.Brightness(image)
            brightness = random.choice([0.5, 0.8, 1.2, 1.5, 1.8])
            image = enh_bri.enhance(brightness)
            # image.show()

        # 色度增强
        if random.choice([0, 1]):
            enh_col = ImageEnhance.Color(image)
            color = random.choice([0.5, 0.8, 1.2, 1.5, 1.8])
            image = enh_col.enhance(color)
            # image.show()

        # 对比度增强
        if random.choice([0, 1]):
            enh_con = ImageEnhance.Contrast(image)
            contrast = random.choice([0.5, 0.8, 1.2, 1.5, 1.8])
            image = enh_con.enhance(contrast)
            # image.show()

        # 锐度增强
        if random.choice([0, 1]):
            enh_sha = ImageEnhance.Sharpness(image)
            sharpness = random.choice([0.5, 0.8, 1.2, 1.5, 1.8])
            image = enh_sha.enhance(sharpness)
            # image.show()

        # mo hu
        if random.choice([0, 0, 1, 0, 0]):
            image = image.filter(ImageFilter.BLUR)
        img = np.asarray(image)
        return img

    # ----------------------- Batch Generator ----------------------------------

    def _aux_generator(self, batch_size=16, stacks=4, normalize=True, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            train_img = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
            train_weights = np.zeros((batch_size, len(self.joints_list)), np.float32)
            i = 0
            color_list = ['RGB', 'BGR', 'HSV']
            while i < batch_size:
                try:
                    if sample_set == 'train':
                        name = random.choice(self.train_set)
                    elif sample_set == 'valid':
                        name = random.choice(self.valid_set)
                    joints = self.data_dict[name]['joints']
                    # box = self.data_dict[name]['box']
                    weight = np.asarray(self.data_dict[name]['weights'])
                    color = random.choice(color_list)
                    img = self.open_img(name, color=color)
                    padd, cbox = self._crop_data_new(img.shape[0], img.shape[1])
                    new_j = self._relative_joints(cbox, padd, joints, to_size=64)
                    hm = self._generate_hm(64, 64, new_j, 64, weight)
                    img = self._crop_img(img, padd, cbox)
                    img = img.astype(np.uint8)
                    img = scm.imresize(img, (256, 256))
                    # image augmentation
                    img, weight = self._random_erase(img, new_j, weight, probability=0.8)
                    img = self._color_augment(img)
                    img, hm = self._rotate_augment(img, hm)

                    hm = np.expand_dims(hm, axis=0)
                    hm = np.repeat(hm, stacks, axis=0)
                    train_weights[i] = weight
                    if normalize:
                        train_img[i] = img.astype(np.float32) / 255
                    else:
                        train_img[i] = img.astype(np.float32)
                    train_gtmap[i] = hm
                    i = i + 1
                except:
                    print('error file: ', name)
            yield train_img, train_gtmap, train_weights

    def generator(self, batchSize=16, stacks=4, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)

    # ---------------------------- Image Reader --------------------------------
    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        # print('dir:', os.path.join(self.img_dir, name))
        img = cv2.imread(os.path.join(self.img_dir, name))
        # print(os.path.join(self.img_dir, name))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def open_resize(self, name, output_size=256, color='RGB'):
        """ Open an image and crop it to given size
               Args:
                   name	: Name of the sample
                   color	: Color Mode (RGB/BGR/GRAY)
               """
        img = self.open_img(name, color=color)
        height = img.shape[0]
        width = img.shape[1]
        img = img.astype(np.uint8)
        padding = [[0, 0], [0, 0], [0, 0]]
        size_rate = width/output_size
        if width == height:
            pass
        elif width > height:
            pad_size = (width - height)//2
            padding[0][0] = padding[0][1] = pad_size
        else:
            pad_size = (height - width)//2
            padding[1][0] = padding[1][1] = pad_size
            size_rate = height/output_size
        img = np.pad(img, padding, mode='constant')
        img = scm.imresize(img, (output_size, output_size))
        return img.astype(np.float32), padding, size_rate

    def plot_img(self, name, plot='cv2'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()

    # ---------------------------- Heatmap Processor --------------------------------

    def normalize_and_find_nkeypoints(self, heatmaps):
        """
        normalize heatmaps and find the correspond max keypoints
        :param heatmaps:
        :param n_max: the number of keypoints you want find from one heatmap
        :return: normalized heatmaps, keypoints
        """
        keypoints = []
        for i in range(heatmaps.shape[0]):
            min_point = np.min(heatmaps[i])
            heatmaps[i] = heatmaps[i] - min_point
            max_point = np.max(heatmaps[i])
            # print('min_max_point of ', i, 'heatmap:', min_point, '   ', max_point)
            heatmaps[i] = heatmaps[i] / max_point * 255
            maxpoint = np.argmax(heatmaps[i])
            col = heatmaps.shape[2]
            keypoint = np.array([maxpoint % col, maxpoint / col])
            keypoints.append(keypoint)
        return heatmaps, np.array(keypoints).astype(np.uint32)

    def find_nkeypoints(self, heatmaps, n_max = 1):
        """
        find the correspond n max keypoints, for now n = 1
        :param heatmaps:
        :param n_max: the number of keypoints you want find from one heatmap
        :return: keypoints
        """
        keypoints = []
        for i in range(heatmaps.shape[0]):
            maxpoint = np.argmax(heatmaps[i])
            col = heatmaps.shape[2]
            keypoint = np.array([maxpoint % col, maxpoint / col])
            # keypoint = divmod(int(keypoint), heatmaps[i].shape[1]) #这样做出来x y轴相反
            keypoints.append(keypoint)
        return np.array(keypoints).astype(np.uint32)

    def print_key_matrix(self, heatmaps, keypoints, matrix_size = 5):
        """
        :param heatmaps:
        :param keypoints:
        :return: print the matrix values around keypoints
        """
        width = heatmaps.shape[2]
        height = heatmaps.shape[1]
        half_matrix = matrix_size // 2
        for i in range(heatmaps.shape[0]):
            keypoint = keypoints[i]
            heatmap = heatmaps[i]
            print('matrix: ', i)
            print(' max ', heatmap[keypoint[1], keypoint[0]])
            if keypoint[0] - half_matrix < 0:
                keypoint[0] = half_matrix
            if keypoint[1] - half_matrix < 0:
                keypoint[1] = half_matrix
            if keypoint[0] + half_matrix >= width:
                keypoint[0] = width - half_matrix - 1
            if keypoint[1] + half_matrix >= height:
                keypoint[1] = height - half_matrix - 1
            left_up = keypoint - half_matrix
            for k in range(matrix_size):
                for p in range(matrix_size):
                    print(heatmap[left_up[1]+k][left_up[0]+p], end=' ')
                print(' ')

    def restore_heatmap(self, heatmaps, padding, size_rate):
        """
        restore all heatmaps to image size
        :param heatmaps: input heatmaps
        :param padding: padding size
        :param size_rate: size rate
        :return:new_heatmap
        """
        new_heatmap = []
        heatmap_size = heatmaps[0].shape[1]
        re_size = int(heatmap_size*size_rate*4)
        # print('shape of heatmap is', heatmaps.shape)
        for i in range(heatmaps.shape[0]):
            heatmaps[i] = heatmaps[i].astype(np.uint8)
            heatmap = scm.imresize(heatmaps[i], (re_size, re_size), interp='cubic')
            heatmap = heatmap[padding[0][0]:re_size-padding[0][1], padding[1][0]:re_size-padding[1][1]]
            new_heatmap.append(heatmap)
        return np.array(new_heatmap).astype(np.float32)

    def test(self, toWait=0.2):
        """ TESTING METHOD
        You can run it to see if the preprocessing is well done.
        Wait few seconds for loading, then diaporama appears with image and highlighted joints
        /!\ Use Esc to quit
        Args:
            toWait : In sec, time between pictures
        """
        self._read_train_data()
        self._create_sets()
        for i in range(50):
            img = self.open_img(self.train_set[i])
            w = self.data_dict[self.train_set[i]]['weights']
            padd, box = self._crop_data_new(img.shape[0], img.shape[1])
            new_j = self._relative_joints(box, padd, self.data_dict[self.train_set[i]]['joints'], to_size=256)
            rhm = self._generate_hm(256, 256, new_j, 256, w)
            rimg = self._crop_img(img, padd, box)
            rimg = scm.imresize(rimg, (256, 256))
            new_img, new_w = self._random_erase(rimg, new_j, w, probability=1)
            grimg = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
            cv2.imshow('image', grimg / 255 + np.sum(rhm, axis=2, ))
            print('joint:', new_j)
            print('weight:', new_w)
            # Wait
            time.sleep(toWait)
            if cv2.waitKey(1000000) == 27:
                print('Ended')
                cv2.destroyAllWindows()
                break







