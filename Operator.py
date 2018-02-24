import utils as u
import tensorflow as tf
import load as l
import numpy as np
from tqdm import tqdm


class Operator:
    def __init__(self, sess, model, config):
        self.model = model
        self.config = config
        self.sess = sess

        #initialize local and global variables
        self.init = tf.group(tf.global_variables_initializer() ,tf.local_variables_initializer())

        # initialize all variables in graph
        self.sess.run(self.init)

        # for saving and restoring model
        self.saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        self.saver_best = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

        # load last check point and continue training
        self.load_model()

        ##############################################

        # initialize summaries for tensor board

        #values to log during process
        self.scalar_summary_tags= ['train_loss_per_epoch','train_acc_per_epoch', 'val_loss_per_epoch', 'val_acc_per_epoch']

        #each value we want to log in tensorboard has a tag placeholder and an operation to write it
        self.summary_tags=[]
        self.summary_placeholders={}
        self.summary_ops={}

        #initialize summaries
        self.init_summaries()

        #create summary writer

        self.summary_writer = tf.summary.FileWriter(self.config.summaries_dir)
        ########################################

        # load training, test and validation data.
        self.load_data()


        # summary functionality

    def init_summaries(self):
        """
        for each tag add a placeholder and operation
        """
        with tf.variable_scope("train_summary_per_epoch"):
            for tag in self.scalar_summary_tags:
                self.summary_tags+=tag
                self.summary_placeholders[tag]=tf.placeholder(tf.float32,name=tag)
                self.summary_ops[tag]=tf.summary.scalar(tag,self.summary_placeholders[tag])


    def add_summary(self, step ,summaries_dict=None, summaries_merged=None):
        """
            log values calculated manually each epoch or calculated from graph each step
        """
        if summaries_dict is not None:

            #run summary op for each tag and value in the dictionary
            summary_list=self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                      {self.summary_placeholders[tag]:value for tag,value in summaries_dict.items()})
            #write in tensor board
            for summ in summary_list:
                self.summary_writer.add_summary(summ,step)

        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged,step)

        pass



    def load_model(self):
        """
        load latest check point to continue training
        :return:

        """
        checkpoint_dir = self.config.checkpoint_dir
        print("searching for latest checkpoint..")
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("loading last checkpoint...{} ".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("model loaded from latest checkpoint \n")
        else:
            print("no checkpoints found, training for the first time")

    def load_best_model(self):
        """
        loads model with best accuracy for testing
        :return:
        """
        checkpoint_dir = self.config.checkpoint_best_dir
        print("searching for best checkpoint..")
        latest_chekpoint = tf.train.latest_checkpoint(checkpoint_dir)

        if latest_chekpoint:
            print("loading last checkpoint...{} ".format(latest_chekpoint))
            self.saver.restore(self.sess, latest_chekpoint)
            print("best checkpoint loaded...\n")
        else:
            print("no best checkpoint")
            exit(-1)

    def save_model(self):
        """
        saves model checkpoint
        :return:
        """
        exp_dir = self.config['checkpoint_dir']

        print("saving check point..")
        self.saver.save(self.sess, exp_dir, self.model.global_step)

        print("saved checkpoint successfully..\n")

    def save_best_model(self):
        """
        saves model checkpoint with best accuracy
        :return:
        """
        best_dir = self.config['checkpoint_best_dir']

        print("saving check point..")
        self.saver.save(self.sess, best_dir, self.model.global_step)

        print("saved checkpoint successfully..\n")

    def load_data(self):
        data_dir = self.config.data_dir
        batch_size = self.config.batch_size
        x_train, y_train, x_val, y_val, x_test, y_test = l.load_dataset(data_dir)
        print("data loaded successfully...")

        # number of iterations to go through entire training set
        self.train_data = {'x': x_train, 'y': y_train}
        self.train_iterations_per_epoch = (x_train.shape[0] + batch_size - 1) // batch_size

        print("x_training shape : ", x_train.shape[0])
        print("y_training shape : ", y_train.shape[0])
        print("num of iterations on training data in one epoch : ", self.train_iterations_per_epoch)

        #####################################################

        self.val_data = {'x': x_val, 'y': y_val}

        self.val_iterations_per_epoch = (x_val.shape[0] + batch_size - 1) // batch_size

        print("x_validation shape : ", x_val.shape[0])
        print("y_validation shape : ", y_val.shape[0])
        print("num of iterations on validation data in one epoch : ", self.val_iterations_per_epoch)

        ########################################################

        self.test_data = {'x': x_test, 'y': y_test}

        # iterations to go through test data, +1 if data size not divisible by batch size
        self.test_iterations_per_epoch = (x_test.shape[0] + batch_size - 1) // batch_size

        print("x_test shape : ", x_test.shape[0])
        print("y_test shape : ", y_test.shape[0])
        print("num of iterations on test data in one epoch : ", self.test_iterations_per_epoch)

        print("data loading complete ...\n")

    def generator(self):

        start = 0
        new_epoch_flag = True
        while True:

            if new_epoch_flag:
                idx = np.random.choice(self.train_iterations_per_epoch, self.train_iterations_per_epoch, replace=False)
                new_epoch_flag = False

            mask = idx[start:start + self.config.batch_size]
            x_batch = self.train_data['x'][mask]
            y_batch = self.train_data['y'][mask]

            start += self.config.batch_size

            yield x_batch, y_batch

            if start >= self.train_data['x'].shape[0]:
                start = 0
                new_epoch_flag = True
                return

    def val(self):

        # current iteration and epoch
        step = self.model.global_step.eval(session=self.sess)
        epoch = self.model.global_epoch_tensor.eval(session=self.sess) - 1  # epoch number was incremented before validation

        print("Validation at step:" + str(step) + " at epoch:" + str(epoch) + " ..")

        # best validation accuracy so far
        best_val = self.model.best_acc_tensor.eval(session=self.sess)
        start = 0
        loss_list = []
        acc_list = []
        for it in tqdm(range(self.val_iterations_per_epoch), total=self.val_iterations_per_epoch, desc='Val-epoch--'):

            x_batch = self.val_data['x'][start:start + self.config.batch_size]
            y_batch = self.val_data['y'][start:start + self.config.batch_size]

            start += self.config.batch_size

            feed = {
                self.model.x: x_batch,
                self.model.y: y_batch,
                self.model.training: False

            }

            loss, acc = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed)
            loss_list += loss
            acc_list += acc

            if start > self.val_data['x'].shape[0]:
                break

        # get avg loss and acc over batches
        total_loss = np.mean(loss_list)
        total_acc = np.mean(acc_list)

        # add to summaries

        # log them to tensor board
        summaries_dict={
            'val_acc_per_epoch':total_acc,
            'val_loss_per_epoch':total_loss
        }

        self.add_summary(step,summaries_dict)

        #print results

        print("Val-epoch-" + str(epoch) + "-" + "loss:" + str(total_loss) + "-" +"acc:" + str(total_acc)[:6])

    # check if best accuracy is achieved
        if total_acc > best_val:
            print("New best accuracy, saving model")
            self.save_best_model()

            # update best accuracy tensor
            self.sess.run(self.model.best_acc_assignop, feed_dict={self.model.best_acc_input: total_acc})

    def test(self):

        print("testing best model..")

        start = 0
        loss_list = []
        acc_list = []
        for it in tqdm(range(self.test_iterations_per_epoch), total=self.test_iterations_per_epoch, desc='test-epoch--'):

            x_batch = self.test_data['x'][start:start + self.config.batch_size]
            y_batch = self.test_data['y'][start:start + self.config.batch_size]

            start += self.config.batch_size

            feed = {
                self.model.x: x_batch,
                self.model.y: y_batch,
                self.model.training: False

            }

            loss, acc = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed)
            loss_list += loss
            acc_list += acc

            if start > self.test_data['x'].shape[0]:
                break

        # get avg loss and acc over batches
        total_loss = np.mean(loss_list)
        total_acc = np.mean(acc_list)

        print("Test statistics")
        print("Total_loss: " + str(total_loss))
        print("Total_acc: " + str(total_acc)[:6])

    def train(self):
        print("start of training process")

        for cur_epoch in range(self.model.global_epoch_tensor.eval(session=self.sess) + 1, self.config.num_epochs + 1, 1):

            # initialize progress bar
            tt = tqdm(self.generator(), total=self.train_iterations_per_epoch, desc='epoch-' + str(cur_epoch))

            # accumulate values to add to summary each epoch
            loss_list = []
            acc_list = []
            summaries_merged = None

            for x_batch, y_batch in tt:
                # prepare feed dict
                feed_dict = {
                    self.model.x: x_batch,
                    self.model.y: y_batch,
                    self.model.training: True
                }

                # run training step
                _, loss, acc, summaries_merged = self.sess.run(
                    [self.model.train_op, self.model.loss, self.model.accuracy,
                     self.model.merged_summary], feed_dict=feed_dict)

                loss_list += loss
                acc_list += acc

                cur_it = self.model.global_step.eval(session=self.sess)

                # add summary per training step
                self.add_summary(cur_it,None,summaries_merged)

            #####################################3
            # end of epoch , add summaries
            total_loss = np.mean(loss_list)
            total_acc = np.mean(acc_list)

            # log them to tensor board

            summaries_dict={
                'train_acc_per_epoch':total_acc,
                'train_loss_per_epoch':total_loss
            }

            self.add_summary(self.model.global_step.eval(session=self.sess),summaries_dict)


            # update epoch number
            self.model.global_epoch_assignop.eval(session=self.sess)

            tt.close()
            print("epoch-" + str(cur_epoch) + "-" + "loss:" + str(total_loss) + "-" + " acc:" + str(total_acc)[:6])

            # save modle after epoch
            self.save_model()

            # perform validation
            self.val()
