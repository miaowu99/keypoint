"""
test main
"""
from dataProcess import Dataprocessor
import configparser


def process_config(conf_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'blouse':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params


if __name__ == '__main__':
    print('--Parsing Config File')
    params = process_config('config.cfg')
    print('--Creating Dataset')
    data_processor = Dataprocessor(params['dress_type'], params['joint_list_blouse'], params['img_directory'], params['training_data_file'])
    dataset = data_processor._read_train_data()[1]
    print(dataset['Images/blouse/0bc88c088232af8d64feea096dac98ae.jpg'])
