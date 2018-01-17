import tensorflow as tf


class Network(object):
    """
    Network class, includes implementation of cnn layers. 
    """
    def __init__(self, inputs, keep_p, upsize = None, is_train=True, trainable=True):
        """
        Initialization
        """
        self.inputs = inputs
        self.keep_p = keep_p
        self.upsize = upsize
        self.is_train = is_train
        self.trainable = trainable
        self.setup() 
 
    def setup(self):
        """
        This function defines the cnn model,
        need to be implemented in subclasses
        """
        raise NotImplementedError("Need to be implemented in subclasses")

    @staticmethod
    def conv(x,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu,
             group=1,
             padding="SAME",
             trainable=True):
        """
        Function for convolutional layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- c_o: Amount of kernels
        --- s_h: Stride in height
        --- s_w: Stride in width
        --- name: Layer name
        --- relu: Do relu or not
        --- group: Amount of groups
        --- padding: Padding method, SAME or VALID
        --- trainable: Whether the parameters in this layer are trainable
        Output:
        --- outputs: Output of the convolutional layer
        """
        with tf.name_scope(name), tf.variable_scope(name):
            # Get the input channel
            c_i = x.get_shape()[-1]/group
            # Create the weights, with shape [k_h, k_w, c_i, c_o]
            weights = tf.get_variable("weights", [k_h, k_w, c_i, c_o], trainable=trainable)
            # Create the biases, with shape [c_o]
            biases = tf.get_variable("biases", [c_o], initializer=tf.zeros_initializer, trainable=trainable)
            # Create a function for convolution calculation
            def conv2d(i, w):
                return tf.nn.conv2d(i, w, [1, s_h, s_w, 1], padding)
            # If we don't need to divide this convolutional layer
            if group == 1:
                outputs = conv2d(x, weights)
            # If we need to divide this convolutional layer
            else:
                # Split the input and weights
                group_inputs = tf.split(x, group, 3, name="split_inputs")
                group_weights = tf.split(weights, group, 3, name="split_weights")
                group_outputs = [conv2d(i, w) for i, w in zip(group_inputs, group_weights)]
                # Concatenate the groups
                outputs = tf.concat(group_outputs, 3)
            # Add the biases
            outputs = tf.nn.bias_add(outputs, biases)
            if relu:
                # Nonlinear process
                outputs = tf.nn.relu(outputs)
            # Return layer's output
            return outputs

    @staticmethod
    def max_pool(x,
                 k_h,
                 k_w,
                 s_h,
                 s_w,
                 name,
                 padding="VALID"):
        """
        Function for max pooling layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- s_h: Stride in height
        --- s_w: Stride in width
        --- name: Layer name
        --- padding: Padding method, SAME or VALID
        Output:
        --- outputs: Output of the max pooling layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.max_pool(x, [1, k_h, k_w, 1], [1, s_h, s_w, 1], padding)
            # Return layer's output
            return outputs

    @staticmethod
    def relu(x, name):
        """
        Function for relu layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        Output:
        --- outputs: Output of the relu layer
        """

        with tf.name_scope(name):
            outputs = tf.nn.relu(x)
            # Return layer's output
            return layer_output

    @staticmethod
    def fc(x, nout, name, relu, trainable=True):
        """
        Function for fully connected layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- nout: Amount of output neurals
        --- name: Layer name
        --- relu: Do relu or not
        --- trainable: Whether the parameters in this layer are trainable
        Output:
        --- outputs: Output of the fc layer
        """

        with tf.name_scope(name), tf.variable_scope(name):
            # Reshape the input
            input_shape = x.get_shape()
            # If the input is 4-D, reshape it to 2-D
            if len(input_shape) == 4:
                dim = 1
                for d in input_shape.as_list()[1:]:
                    dim *= d
                x = tf.reshape(x, [-1, dim])
            else:
                dim = input_shape.as_list()[1]
            # Get the weights, with shape [dim, nout]
            weights = tf.get_variable("weights", [dim, nout], trainable=trainable)
            # Get the biases, with shape [nout]
            biases = tf.get_variable("biases", [nout], initializer=tf.zeros_initializer, trainable=trainable)
            # Calculate the output of layer
            outputs = tf.nn.xw_plus_b(x, weights, biases)
            if relu:
                outputs = tf.nn.relu(outputs)
            # Return layer's output
            return outputs

    @staticmethod
    def lrn(x,
            name,
            k=1.0,
            n=2,
            alpha=2e-05,
            beta=0.75):
        """
        Function for local response normalization layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        --- rest: Parameters for lrn layer
        Output:
        --- outputs: Output of the lrn layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.local_response_normalization(x, n, k, alpha, beta)
            # Return layer's output
            return outputs

    @staticmethod
    def dropout(x, keep_prob, name):
        """
        Function for dropout layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- keep_prob: Keep probability for dropout layer
        --- name: Layer name
        Output:
        --- outputs: Output of the dropout layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.dropout(x, keep_prob)
            # Return layer's output
            return outputs

    @staticmethod
    def concat(x, dim, name):
        """
        Function for concat layer

        Input:
        --- x: Layer input, list of 4-D Tensors, each with shape [bsize, height, width, channel]
        --- dim: Which dimension to concat
        --- name: Layer name
        Output:
        --- outputs: Output of the concat layer
        """
        with tf.name_scope(name):
            outputs = tf.concat(x, dim)
            # Return layer's output
            return outputs

    def upsample(self, x, name):
        """
        Function for upsample layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        Output:
        --- outputs: Output of the upsample layer
        """
        with tf.name_scope(name):
            if self.upsize is not None:
                size = self.upsize
            else:
                size = self.inputs.get_shape().as_list()[1:3]
            outputs = tf.image.resize_bilinear(x, size)
            # Return layer's output
            return outputs

    @staticmethod
    def softmax(x, name):
        """
        Function for softmax layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- name: Layer name
        Output:
        --- outputs: Output of the softmax layer
        """
        with tf.name_scope(name):
            outputs = tf.nn.softmax (x)
            # Return layer's output
            return outputs
  
    @staticmethod      
    def atrous_conv(x,
                    k_h,
                    k_w,
                    c_o,
                    rate,
                    name,
                    relu,
                    padding="SAME",
                    trainable=True):
        """
        Function for atrous convolutional layer

        Input:
        --- x: Layer input, 4-D Tensor, with shape [bsize, height, width, channel]
        --- k_h: Height of kernels
        --- k_w: Width of kernels
        --- c_o: Amount of kernels
        --- rate: Rate of dilation
        --- name: Layer name
        --- relu: Do relu or not
        --- padding: Padding method, SAME or VALID
        --- trainable: Whether the parameters in this layer are trainable
        Output:
        --- outputs: Output of the atrous convolutional layer
        """
        with tf.name_scope(name), tf.variable_scope(name):
            # Get the input channel
            c_i = layer_input.get_shape()[-1]
            # Create the weights, with shape [k_h, k_w, c_i, c_o]
            weights = tf.get_variable("weights", [k_h, k_w, c_i, c_o], trainable=trainable)
            # Create the biases, with shape [c_o]
            biases = tf.get_variable("biases", [c_o], initializer=tf.zeros_initializer, trainable=trainable)
            # Do atrous conv
            outputs = tf.nn.atrous_conv2d(x, weights, rate, padding)
            # Add the biases
            outputs = tf.nn.bias_add(outputs, biases)
            if relu:
                # Nonlinear process
                outputs = tf.nn.relu(outputs)
            # Return layer's output
            return outputs
	
    
