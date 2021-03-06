"""
TRAIN LAUNCHER 

"""

import configparser
from hourglass_tiny import HourglassModel
from dataProcess import DataGenerator


def process_config(conf_file, test_type = None):
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
		if test_type == None:
			if section == params['dress_type']:
				for option in config.options(section):
					params[option] = eval(config.get(section, option))	
		else:
			if section == test_type:
				for option in config.options(section):
					params[option] = eval(config.get(section, option))	
	return params


if __name__ == '__main__':
	print('--Parsing Config File')
	params = process_config('config.cfg')

	print('--Creating Dataset:',params['dress_type'])
	dataset = DataGenerator(params['dress_type'], params['joint_list'], params['train_img_directory'], params['training_data_file'])
	dataset.creator()
	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'],
                           outputDim=params['num_joints'], batch_size=params['batch_size'], attention=params['mcam'], training=True,
                           drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'],
                           decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'],
                           logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'],
                           joints=params['joint_list'], modif=False)
	model.generate_model()
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], saver_dir=params['model_file'], dataset=None, load=params['model_file'] + params['load_file'])
