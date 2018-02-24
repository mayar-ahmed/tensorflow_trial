import numpy as np
import yaml
import os

from easydict import EasyDict as edict

def read_args():
    #read arguments in experiment file and return a dictionary
    config = {}

    with open('exp.yaml', 'r') as fp:
        config.update(yaml.load(fp))

    ('--------------------experiment arguments------------------')
    for k in config.keys():
        print("{}: {}".format(k, config[k]))

    print("---------------------------------------------------------\n")

    return  edict(config)

########################################################################


def create_dirs(dirs):
    #create a directory for each one in dirs

    try:

        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)
    except Exception as err:
        print("creating directory failed : {0}".format(err))
        exit(-1)

#########################################################################

def create_exp_dirs(config):
    #create checkpoints, best checkpoint and summary directories

    config['exp_dir'] = os.path.realpath(os.getcwd()) + "/experiments/" + config.exp_dir
    exp_dir=config.exp_dir
    config['checkpoint_dir']=exp_dir+'checkpoints/'
    config['checkpoint_best_dir']=exp_dir + 'checkpoints/best/'
    config['summaries_dir']=exp_dir + 'summaries/'




    create_dirs([exp_dir, config.checkpoint_dir,config.checkpoint_best_dir,config.summaries_dir])

    return config



############################################################################3