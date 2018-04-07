"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_tiny import HourglassModel
from dataProcess import DataGenerator
import cv2
import numpy as np


def process_config(conf_file):
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
		if section == params['dress_type']:
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config('config.cfg')

	print('--Creating Dataset')
	dataset = DataGenerator(params['dress_type'], params['joint_list'], params['train_img_directory'], test_data_file=params['training_data_file'])
	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'],
                           outputDim=params['num_joints'], batch_size=params['batch_size'], attention=params['mcam'], training=True,
                           drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],
                           decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'],
                           logdir_test=params['log_dir_test'], tiny=params['tiny'], w_loss=params['weighted_loss'],
                           joints=params['joint_list'], modif=False)
	model.generate_model()
	model.restore(load=params['model_file'] + params['load_file'])
	test_table = dataset.read_test_data()
	for i in range(len(test_table)):
		image = dataset.open_img(test_table[i])
		output, keypoints = model.get_output(test_table[i])
		print('image_width', image.shape[1], '   image_height', image.shape[0])
		print('keypoints:', end='       ')
		for p in range(keypoints.shape[0]):
			print(keypoints[p], end='  ')
		print(' ')
		for k in range(keypoints.shape[0]):
			# cv2.imshow(str('point' + str(k)), output[k])
			point = tuple(keypoints[k])
			cv2.circle(image, point, 5, (k/keypoints.shape[0]*255, (1 - k/keypoints.shape[0])*255, 255), -1)

		output = output.astype(np.uint8)
		cv2.imshow('image', image)
		# Wait
		if cv2.waitKey(2000000) == 27:
			print('Ended')
			cv2.destroyAllWindows()
			break


