import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

from model import CaffeFCN
from image_reader import Reader
from utils import decode_labels, inv_preprocess

# Default values
MODE = "train"
RESTORE = False

# Configures 
bsize = 20
nclass = 40
lr_base = 1
momentum = 0.9
weight_decay = 0.0005
training_steps = 25000
image_size = (480, 640)
img_mean = [116.190, 97.203, 92.318]

# Dirs
ckpt_dir = "./ckpts"
data_list_dir = "./Data/NYUD_v2/"

class CaffeFCNSegmentation(object):
    """
    CaffeFCNSegmentation class definition
    """
    def __init__(self, param_path="DEFAULT"):
        # Get the path of the parameters
        if param_path == "DEFAULT":
            self.param_path = "./CaffeNet/bvlc_caffenet.npy"
        else:
            self.param_path = param_path
        # Get the arguments
        self.args = self.get_arguments()

    def save(self, saver, sess, global_step):
        """
        Function for saving the model
        """
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
        saver.save(sess, ckpt_path, global_step)

    def load(self, loader, sess):
        """
        Function for loading the model
        """
        if os.path.exists(ckpt_dir):
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                loader.restore(sess, ckpt.model_checkpoint_path)


    def get_trained_params(self, sess):
        """
        Function to get trained parameters from the param_path
        """
        # If the param_path exists
        if os.path.exists(self.param_path):
            # Load the parameter dictionary
            # Form of the dictionary:
            # {"layer_name_1":{"weights":[......],"biases":[......]},
            #  "layer_name_2":{"weights":[......],"biases":[......]},
            #  ...............................................,
            #  "layer_name_n":{"weights":[......],"biases":[......]}}
            param_dict = np.load(self.param_path, encoding='bytes').item()
            # Restore layers
            for layer_name in param_dict:
                if 'fc' not in layer_name:
                    with tf.variable_scope('net_a/'+layer_name, reuse=True):
                        for param_name, data in param_dict[layer_name].iteritems():
                            var = tf.get_variable(param_name)
                            sess.run(var.assign(data))
                elif layer_name == 'fc6':
                    with tf.variable_scope('net_a/'+layer_name, reuse=True):
                            fc6_weight = param_dict['fc6']['weights']
                            fc6_weight = np.reshape(fc6_weight, [6, 6, 256, 4096])
                            var = tf.get_variable('weights')
                            sess.run(var.assign(fc6_weight))
                            fc6_bias = param_dict['fc6']['biases']
                            var = tf.get_variable('biases')
                            sess.run(var.assign(fc6_bias))
                elif layer_name == 'fc7':
                    with tf.variable_scope('net_a/'+layer_name, reuse=True):
                            fc7_weight = param_dict['fc7']['weights']
                            fc7_weight = np.reshape(fc7_weight, [1, 1, 4096, 4096])
                            var = tf.get_variable('weights')
                            sess.run(var.assign(fc7_weight))
                            fc7_bias = param_dict['fc7']['biases']
                            var = tf.get_variable('biases')
                            sess.run(var.assign(fc7_bias))

    def train(self):
        """
        Function to train the model
        """
        # Create queue coordinator.
        coord = tf.train.Coordinator()
        # Set the graph-level random seed
        tf.set_random_seed(1234)
        # Read the images and the labels
        with tf.name_scope("create_inputs"):
            data_list = os.path.join(data_list_dir, "train.txt")
            train_reader = Reader(coord, data_list, True, True, True, True)
            img_bat, lab_bat = train_reader.dequeue(bsize)
        # --Inference:
        # Create the CaffeFCN network
        with tf.name_scope("net_a"), tf.variable_scope("net_a"):
            net = CaffeFCN(img_bat, keep_p=[0.2, 0.5])
        # Get the output of the network
        raw_preds = net.score_up
        # Get the prediction
        preds = tf.argmax(raw_preds, axis=3)
        preds = tf.expand_dims(preds, dim=3)
        # --Define the loss function:
        # Get the weights of layers
        weight_list = [w for w in tf.trainable_variables() if "weights" in w.name]
        with tf.name_scope("loss"):
            with tf.name_scope("reg_loss"):
                # Get the reg loss
                reg_loss = [weight_decay * tf.nn.l2_loss(w) for w in weight_list]
                reg_loss = tf.add_n(reg_loss, "reg_loss")
            with tf.name_scope("data_loss"):
                # Get the data loss
                # Flatten the preds and labels
                flat_preds = tf.reshape(raw_preds, [-1, nclass])
                flat_labels = tf.reshape(lab_bat, [-1, ])
                indices = tf.squeeze(tf.where(tf.less_equal(flat_labels, nclass - 1)), 1)
                flat_preds = tf.gather(flat_preds, indices)
                flat_labels = tf.cast(tf.gather(flat_labels, indices), tf.int32)
                data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_labels, logits=flat_preds)
                data_loss = tf.reduce_mean(data_loss, name="data_loss")
            # Get the total loss
            loss = tf.add(data_loss, reg_loss, "total_loss")
        with tf.name_scope("optimizer"):
            # Define the optimizer:
            # Create a global step variable
            global_step = tf.Variable(0, trainable=False, name="global_step")
            # Create the optimizer objects
            p1_opt = tf.train.MomentumOptimizer(self.args.learning_rate_base * 0.001, self.args.momentum)
            p2_opt = tf.train.MomentumOptimizer(self.args.learning_rate_base * 0.002, self.args.momentum)
            # Get the variables
            p1 = [p for p in tf.trainable_variables() if "score_fr/biases" not in p.name]
            p2 = [p for p in tf.trainable_variables() if "score_fr/biases" in p.name]
            # Get the gradients
            grads = tf.gradients(loss, p1 + p2)
            grads_vals = zip(grads, p1 + p2)
            grads_vals_p1 = grads_vals[ : len(p1)]
            grads_vals_p2 = grads_vals[len(p1) : len(p1) + len(p2)]
            update_p1 = p1_opt.apply_gradients(grads_vals_p1, global_step)
            update_p2 = p2_opt.apply_gradients(grads_vals_p1)
            update_op = tf.group(update_p1, update_p2)
        # Summary
        with tf.name_scope("summary"):
            # loss summary
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("reg_loss", reg_loss)
            tf.summary.scalar("data_loss", data_loss)
            # grads and vals summary
            for grad, val in grads_vals:
            tf.summary.histogram(val.name, val)
            tf.summary.histogram(val.name + "_grads", grad)
            # Image summary.
            images_summary = tf.py_func(inv_preprocess, [img_bat, 2, img_mean], tf.uint8)
            labels_summary = tf.py_func(decode_labels, [lab_bat, 2, nclass], tf.uint8)
            preds_summary = tf.py_func(decode_labels, [preds, 2, nclass], tf.uint8)
            tf.summary.image('images', tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), max_outputs=2)
            # Merge
            merge = tf.summary.merge_all()
        # Create Saver objects for save and restore
        saver = tf.train.Saver(max_to_keep=15)
        # Create a initializer
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        # Run the graph in the session
        with tf.Session() as sess:
            # Initialize the variables
            sess.run([init_g, init_l])
            # Open the Tensorboard
            writer = tf.summary.FileWriter("./graphs", graph=sess.graph)
            # Get the trained parameters
            self.get_trained_params(sess)
            if self.args.restore:
                self.load(saver, sess)
            # Start queue threads.
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            # Train the model
            print "{} -- Start Training:".format(datetime.now())
            for step in range(training_steps):
                gstep = global_step.eval(session=sess)
                if (gstep % 100 == 0) and (step != 0):
                    self.save(saver, sess, global_step)
                    lo, m, _ =  sess.run([loss, merge, update])
                    writer.add_summary(m, gstep)
                    print "{0}: After {1} training steps, the loss = {2}".format(datetime.now(), gstep, lo)
                elif gstep % 50 == 0:
                    lo, _ =  sess.run([loss, update])
                    print "{0}: After {1} training steps, the loss = {2}".format(datetime.now(), gstep, lo)
                else:
                    sess.run(update)
            print "{} -- Training Stopped".format(datetime.now())
            coord.request_stop()
            coord.join(threads)
        writer.close()

    def test(self):
        # Create queue coordinator.
        coord = tf.train.Coordinator()
        # Set the graph-level random seed
        tf.set_random_seed(1234)
        # Read the images and the labels
        with tf.name_scope("create_inputs"):
            data_list = os.path.join(data_list_dir, "val.txt")
            test_reader = Reader(coord, data_list)
            img = test_reader.img
            lab = test_reader.lab
            img_bat = tf.expand_dims(img, dim=0)
            lab_bat = tf.expand_dims(lab, dim=0)
        # --Inference:
        # Create the CaffeFCN network
        with tf.name_scope("net_a"), tf.variable_scope("net_a"):
            net = CaffeFCN(img_bat, keep_prob=[1, 1])
        # Get the output of the network
        raw_preds = net.score_up
        # Get the prediction
        preds = tf.argmax(raw_preds, axis=3)
        preds = tf.expand_dims(preds, dim=3)
        # Metrics
        with tf.name_scope("metrics"):
            flat_preds = tf.reshape(preds, [-1, ])
            flat_gt = tf.reshape(lab_bat, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(flat_gt, nclass - 1)), 1)
            flat_preds = tf.gather(flat_preds, indices)
            flat_gt = tf.cast(tf.gather(flat_gt, indices), tf.int32)
            # mIoU
            with tf.name_scope("mIoU"):
                mIoU, mIoU_update = tf.contrib.metrics.streaming_mean_iou(flat_preds, flat_gt, num_classes=nclass)
            # Accuracy
            with tf.name_scope("accuracy"):
                accu, accu_update = tf.contrib.metrics.streaming_accuracy(flat_preds, flat_gt)
            # fwIU
            with tf.name_scope("fwIU"):
                total_cm, fwIU_update = _streaming_confusion_matrix(flat_gt, flat_preds, nclass)
                sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
                sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
                cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
                denominator = sum_over_row + sum_over_col - cm_diag
                denominator = array_ops.where(math_ops.greater(denominator, 0),
                                              denominator,
                                              array_ops.ones_like(denominator))
                iou = math_ops.div(cm_diag, denominator)
                frequency = sum_over_col/math_ops.reduce_sum(sum_over_col)
                fwIU = math_ops.reduce_sum(frequency * iou)
        # Image summary
        with tf.name_scope("summary"):
            images_summary = tf.py_func(inv_preprocess, [img_bat, 1, img_mean], tf.uint8)
            labels_summary = tf.py_func(decode_labels, [lab_bat, 1, nclass], tf.uint8)
            preds_summary = tf.py_func(decode_labels, [raw_pred, 1, nclass], tf.uint8)
            tf.summary.image('images', tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), max_outputs=1)
            merge = tf.summary.merge_all()
        # Create Saver objects for save and restore
        saver = tf.train.Saver()
        # Create a initializer
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run([init_g, init_l])
            writer = tf.summary.FileWriter("./test_graphs", graph=sess.graph)
            # Load weights.
            self.get_trained_params(sess)
            if self.args.restore:
                self.load(saver, sess)
            # Start queue threads.
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            # Iterate over training steps.
            for step in range(654):
                m_, a_, f_ = sess.run([mIoU_update, accu_update, fwIU_update])
            m, a, f = sess.run([mIoU, accu, fwIU])
            print '-----------Metrics-----------'
            print 'Mean IoU: {:.3f}'.format(m)
            print 'Freq IoU: {:.3f}'.format(f)
            print 'Accuracy: {:.3f}'.format(a)
            print '-----------------------------'
            for step in range(654):
                if step % 20 == 0:
                    m = sess.run(merge)
                    writer.add_summary(m, step)
            coord.request_stop()
            coord.join(threads)
        writer.close()

    @staticmethod
    def get_arguments():
        # Create a parser object
        parser = argparse.ArgumentParser()
        # Add arguments
        parser.add_argument("--mode", default=MODE, type=str, help="Mode: train, eval, infer")
        parser.add_argument("--restore", default=RESTORE, type=bool, help="Whether to restore")
        # Parse the arguments
        return parser.parse_args()


def main():
    model = CaffeFCNSegmentation()
    if model.args.mode == 'train':
        model.train()
    elif model.args.mode == 'test':
        model.test()
    else:
        pass


if __name__ == "__main__":
    main()
