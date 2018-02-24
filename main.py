#!/usr/bin/env python.

from Operator import Operator
from utils import *
import tensorflow as tf
from model2 import Classifier
#main file

#read arguments and create directories
c= read_args()

create_exp_dirs(c)

#reset graph
tf.reset_default_graph()
session=tf.Session()

#create model instance
model=Classifier(c)
model.build()


trainer= Operator(session, model, c)

if c.train:
    trainer.train()
    trainer.save_model()

if c.test:
    trainer.load_best_model()
    trainer.test()




