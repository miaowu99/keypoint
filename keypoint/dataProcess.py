# -*- coding: utf-8 -*-

"""

"""
import pandas as pd
import os


class Dataprocessor():
    """
    To process images and labels
    """
    def __init__(self, dress_type, joints_name, img_dir, train_data_file):
        """Initializer
            Args:
            dress_tpye          : Tpye of dress
            joints_name			: List of joints condsidered
            img_dir				: Directory containing every images
            train_data_file		: Text file with training set data

        """
        self.joints_list = joints_name
        self.img_dir = img_dir
        self.train_data_file = train_data_file
        self.dress_type = dress_type
        self.images = os.listdir(img_dir)

    def _read_train_data(self):
        """
        To read labels in csv
        """
        self.train_table = []     # The names of images being trained
        self.data_dict = {}       # The labels of images
        label_file = pd.read_csv( self.train_data_file)
        print('READING LABELS OF TRAIN DATA')
        label_file = label_file[label_file.image_category == self.dress_type]  # Only take the type we want
        for i in range(label_file.shape[0]):
            joints = []
            name = str(label_file.get_value(i, 'image_id'))
            weight = []
            box = []
            for joint_name in self.joints_list:
                joint_value = []
                value = str(label_file.get_value(i, joint_name))
                value = value.split('_')
                # print(value)
                joint_value.append(int(value[0]))
                joint_value.append(int(value[1]))
                joints.append(joint_value)
                if value[2] != '1':
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
            self.data_dict[name] = {'box': box, 'joints': joints, 'weights': weight}
            self.train_table.append(name)
        print('FINISH')
        return [self.train_table, self.data_dict]

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











