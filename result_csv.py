"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_tiny import HourglassModel
from dataProcess import DataGenerator
import cv2
import numpy as np
import pandas as pd


def process_config(conf_file, dress_type):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == dress_type:
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


def result_write(result_file, dress_type):
	print('--Parsing Config File')
	params = process_config('config.cfg', dress_type)
	print('--Creating Dataset')
	img_dir = params['test_img_directory']
	data_file = params['testing_data_file']
	joint_list = params['joint_list']
	dataset = DataGenerator(dress_type, params['joint_list'], img_dir, test_data_file=data_file)
	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
						   nLow=params['nlow'],
						   outputDim=params['num_joints'], batch_size=params['batch_size'], attention=params['mcam'],
						   training=True,
						   drop_rate=params['dropout_rate'], lear_rate=params['learning_rate'],
						   decay=params['learning_rate_decay'],
						   decay_step=params['decay_step'], dataset=dataset, name=params['name'],
						   logdir_train=params['log_dir_train'],
						   logdir_test=params['log_dir_test'], tiny=params['tiny'], w_loss=params['weighted_loss'],
						   joints=joint_list, modif=False)
	model.generate_model()
	model.restore(load=params['model_file'] + params['load_file'])
	print('START')
	for i in range(result_file.shape[0]):
		if result_file.at[i, 'image_category'] == dress_type:  # Only take the type we want
			name = str(result_file.at[i, 'image_id'])
			print('Dating ', i, ' PICTURE', '   name is:', name)
			image = dataset.open_img(name)
			output, keypoint = model.get_output(name)
			heatmaps = np.zeros((10, output.shape[0], output.shape[1], output.shape[2]))
			heatmaps[0] = output
			for j in range(1, 10):
				heatmaps[j], keypoint = model.get_output(name)
			output = dataset.average_heatmaps(heatmaps)
			keypoints = dataset.find_nkeypoints(output)
			for k in range(keypoints.shape[0]):
				position = str(keypoints[k][0]) + '_' + str(keypoints[k][1]) + '_' + '1'
				result_file.loc[i, joint_list[k]] = position
				point = tuple(keypoints[k])
				cv2.circle(image, point, 5, (0, 0, 255), -1)
			cv2.imshow('image', image)
			# Wait
			if cv2.waitKey(10) == 27:
				print('Ended')
				cv2.destroyAllWindows()
				break
	cv2.destroyAllWindows()
	model.close_sess()    # 释放资源
	print('FINISH')


if __name__ == '__main__':
	dress_types = ['blouse']
	result_file = pd.read_csv('test/test_result_4 12.csv')
	for dress_type in dress_types:
		result_write(result_file, dress_type)
	result_file.to_csv("test/test_result_4 12.csv", index=False, na_rep='-1_-1_-1')

