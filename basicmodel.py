import tensorflow as tf

class BasicModel:
    def __init__(self,config):
        self.config=config


    #build graph
    def build(self):

        #floriginal code kan 3amelha place holders
        training=self.config['is_train']
        reg=self.config['reg']
        dropout=self.config['dropout']
        n_classes=self.config['num_classes']
        data_dim=self.config['data_dim']
        lr=self.config['learning_rate']


        self.x=tf.placeholder(tf.float32, [None]+data_dim)
        self.y=tf.placeholder(tf.float32, [None])

        self.conv1=self.conv_bn_relu(self.x,32, training=training,reg=reg)
        self.conv2=self.conv_bn_relu(self.x,32,training=training,reg=reg)
        self.conv3=self.conv_bn_relu(self.x,32,training=training,reg=reg)

        self.f =tf.layers.flatten(self.conv3)

        self.fc1=self.fc_bn_relu(self.f,128,reg=reg,keep_prob=dropout)
        self.fc2=self.fc_bn_relu(self.fc1,64, keep_prob=dropout)
        self.scores=tf.layers.dense(self.fc2,n_classes,kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

        self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores,labels=self.y)) +\
                    tf.losses.get_regularization_loss() #don't know da sa7 wala a3ml tf.reduce su

        self.prob=tf.nn.softmax(self.scores)
        self.predictions=tf.argmax(self.prob,axis=1)
        self.correct_predictions=tf.equal(self.predictions,self.y)
        self.accuracy=tf.reduce_mean(tf.cast(self.correct_predictions,tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)



    #def the layers you want
    def conv_bn_relu(self,inp, nfilters,training,kh=3, kw=3, pad='same', s=(1, 1),reg=0):

        c=tf.layers.conv2d(inp,nfilters,(kh,kw),s,pad,activation=None,use_bias=True,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

        bn=tf.layers.batch_normalization(c,training=training)

        mx=tf.layers.max_pooling2d(bn,pool_size=(2,2),strides=(2,2))

        relu=tf.nn.relu(mx)
        return relu

    def fc_bn_relu(self,inp,nunits,training,keep_prob,reg=0):

        fc=tf.layers.dense(inp,nunits,kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

        bn=tf.layers.batch_normalization(fc,training=training)

        relu=tf.nn.relu(bn)
        drop=tf.layers.dropout(relu, rate=1-keep_prob)


        return drop



