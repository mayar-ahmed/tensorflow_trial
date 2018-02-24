import tensorflow as tf

class Classifier:
    def __init__ (self,config):
        self.config=config

        #iteration number, updated by optimizer
        self.global_step=tf.Variable(0, trainable=False, name='global_step')


        #for restoring epoch number after pausing
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor=tf.Variable(-1, trainable=False)
            #increment epoch
            self.global_epoch_assignop=self.global_epoch_tensor.assign_add(1)

        #best validation accuracy
        with tf.variable_scope('best_acc'):
            self.best_acc_tensor=tf.Variable(0.0, trainable=False)
            self.best_acc_input=tf.placeholder('float32')
            self.best_acc_assignop=self.best_acc_tensor.assign(self.best_acc_input)




    def build(self):

        data_dim=self.config.data_dim
        reg=self.config.reg
        dropout=self.config.dropout
        num_class=self.config.num_classes
        lr=self.config.learning_rate

        print("model parameters : ")
        print("num classes : {} , lr: {} , data_dim: {} ".format(num_class,lr,data_dim))

        #data and training flag
        self.x=tf.placeholder(tf.float32, shape=[None]+ data_dim)
        self.y=tf.placeholder(tf.int64,shape=[None])
        self.training=tf.placeholder(tf.bool)

        #first block fully connected 128,batch normalization, relu
        self.d1= self.fc(self.x,128,reg)
        self.b1=tf.layers.batch_normalization(self.d1,training=self.training)
        self.a1=tf.nn.relu(self.b1)

        #second block fully connected 1024, batch normalization, relu
        self.d2= self.fc(self.a1,1024,reg)
        self.b2=tf.layers.batch_normalization(self.d2,training=self.training)
        self.a2=tf.nn.relu(self.b2)

        #third block fully connected 512, batch normalization, relu
        self.d3= self.fc(self.a2,512,reg)
        self.b3=tf.layers.batch_normalization(self.d2,training=self.training)
        self.a3=tf.nn.relu(self.b3)

        #fourth block fully connected 256, batch normaliztion, relu
        self.d4= self.fc(self.a3,256,reg)
        self.b4=tf.layers.batch_normalization(self.d2,training=self.training)
        self.a4=tf.nn.relu(self.b4)

        self.final=self.fc(self.a4,num_class,reg)


        self.loss=tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.y,logits=self.final))+\
                  tf.losses.get_regularization_loss()


        self.scores=tf.nn.softmax(self.final)
        self.predictions=tf.argmax(self.scores, axis=1)
        self.correct=tf.equal(self.predictions,self.y)
        self.accuracy=tf.reduce_mean(tf.cast(self.correct,tf.float32))

        tf.summary.scalar("loss_per_iteration", self.loss)
        tf.summary.scalar("acc_per_iteration", self.accuracy)
        self.merged_summary=tf.summary.merge_all()

        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op=tf.train.AdamOptimizer(lr).minimize(self.loss,global_step=self.global_step)



    def fc(self,inp, neurons,reg=0.0,act=None):

        fc1=tf.layers.dense(inp,neurons,act,kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
        return fc1

