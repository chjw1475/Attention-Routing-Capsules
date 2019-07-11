import numpy as np
from keras import regularizers, layers, initializers
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Multiply, Dropout
import tensorflow as tf

K.set_image_data_format('channels_last')

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config

class PrimaryCap(layers.Layer):
    def __init__(self, n_channels, dim_capsule, kernel_regularizer=None, **kwargs):
        super(PrimaryCap, self).__init__(**kwargs)

        self.n_channels = n_channels
        self.dim_capsule = dim_capsule
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
    def build(self, input_shape):
        
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.input_num_features = input_shape[3]
        
        self.convW_1 = self.add_weight(shape =[3, 3, self.input_num_features, self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='convW_1',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.bias_1 = self.add_weight(shape=[self.dim_capsule*self.n_channels], initializer='zeros', name='bias_1',trainable=True)
        self.CapsAct_W = self.add_weight(shape =[1, 1, self.dim_capsule, self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='CapsAct_W',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.CapsAct_B = self.add_weight(shape=[self.dim_capsule*self.n_channels], initializer='zeros', name='CapsAct_B',trainable=True)
        
        
    def call(self, inputs):
        conv1s = tf.nn.conv2d(inputs, self.convW_1, strides=[1, 2, 2, 1], padding='SAME')
        conv1s = tf.nn.bias_add(conv1s, self.bias_1)
        conv1s = tf.nn.relu(conv1s)
        conv1s = tf.split(conv1s, self.n_channels, axis=-1)
        
        CapsAct_ws = tf.split(self.CapsAct_W, self.n_channels, axis=-1)
        CapsAct_bs = tf.split(self.CapsAct_B, self.n_channels, axis=-1)
        
        def func(conv1, CapsAct_w, CapsAct_b):
            output = tf.nn.conv2d(conv1, CapsAct_w, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.nn.bias_add(output, CapsAct_b)
            output = tf.expand_dims(output, axis=-1)
            return output
        
        outputs = [func(conv1, CapsAct_w, CapsAct_b) for conv1, CapsAct_w, CapsAct_b in zip(conv1s, CapsAct_ws, CapsAct_bs)]
        outputs = tf.concat(outputs, axis=-1)
        return outputs
        
    def compute_output_shape(self, input_shape):
        return tuple([None, int(self.input_height/2), int(self.input_width/2), self.dim_capsule, self.n_channels])
    
    def get_config(self):
        config = {
            'n_channels': self.n_channels,
            'dim_capsule': self.dim_capsule
        }
        base_config = super(PrimaryCap, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ConvCaps(layers.Layer):
    def __init__(self, n_channels, dim_capsule, decrease_resolution=False, kernel_regularizer=None, **kwargs):
        super(ConvCaps, self).__init__(**kwargs)

        self.n_channels = n_channels
        self.dim_capsule = dim_capsule
        if decrease_resolution == True:
            self.stride = 2
            self.padding = 'VALID'
        else:
            self.stride = 1
            self.padding = 'SAME'
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.input_dim = input_shape[3]
        self.input_ch = input_shape[4]
        
        self.Att_W = self.add_weight(shape=[1, 1, self.dim_capsule, self.input_ch, self.input_ch*self.n_channels],
                                    initializer='glorot_uniform',  name='Att_W',trainable=True)
        self.ConvTrans_W = self.add_weight(shape =[3, 3, self.input_dim, self.input_ch*self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='ConvTrans_W',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.ConvTrans_B = self.add_weight(shape=[self.input_ch*self.dim_capsule*self.n_channels],
                                           initializer='zeros', name='ConvTrans_B',trainable=True)
        self.FeaExt_W = self.add_weight(shape =[3, 3, self.dim_capsule, self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='FeaExt_W',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.FeaExt_B = self.add_weight(shape=[self.dim_capsule*self.n_channels], initializer='zeros', name='FeaExt_B',trainable=True)
        self.CapsAct_W = self.add_weight(shape =[1, 1, self.dim_capsule, self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='CapsAct_W',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.CapsAct_B = self.add_weight(shape=[self.dim_capsule*self.n_channels], initializer='zeros', name='CapsAct_B',trainable=True)
        
        
    def call(self, inputs):
        input_caps = tf.split(inputs, self.input_ch, axis=-1)
        ConvTrans_ws = tf.split(self.ConvTrans_W, self.input_ch, axis=-1)
        ConvTrans_bs = tf.split(self.ConvTrans_B, self.input_ch, axis=-1)
        
        # Convolutional Transform by 3x3 conv 
        conv1s = [tf.nn.conv2d(tf.squeeze(input_cap, axis=-1), ConvTrans_w, strides=[1, self.stride, self.stride, 1], padding='SAME')
                  for input_cap, ConvTrans_w in zip(input_caps, ConvTrans_ws)]
        conv1s = [tf.reshape(tf.nn.bias_add(conv1, ConvTrans_b), [-1, int(self.height/self.stride), int(self.width/self.stride), self.dim_capsule, self.n_channels, 1])
                  for conv1, ConvTrans_b in zip(conv1s, ConvTrans_bs)]
        conv1s = tf.concat(conv1s, axis=-1) 
        conv1s = tf.transpose(conv1s, [0,1,2,3,5,4])
        
        # Att_inputs shape : (n_ch, batch_sz, h, w, dim_cap, input_ch, 1)
        Att_inputs = tf.split(conv1s, self.n_channels, axis=-1)
        Att_ws = tf.split(self.Att_W, self.n_channels, axis=-1)
        FeaExt_ws = tf.split(self.FeaExt_W, self.n_channels, axis=-1)
        FeaExt_bs = tf.split(self.FeaExt_B, self.n_channels, axis=-1)
        CapsAct_ws = tf.split(self.CapsAct_W, self.n_channels, axis=-1)
        CapsAct_bs = tf.split(self.CapsAct_B, self.n_channels, axis=-1)
        
        def func(conv1, Att_w, FeaExt_w, CapsAct_w, FeaExt_b, CapsAct_b) :
            x = tf.squeeze(conv1, axis=-1) #x.shape = (batch_sz, height, width, dim_cap, input_ch)
            
            # Attention Routing
            # attentions shape =(batch_sz, height, width, 1, input_ch)
            attentions = tf.nn.conv3d(x, Att_w, strides=[1, 1, 1, 1, 1], padding='VALID')
            attentions = tf.nn.softmax(attentions, axis=-1)
            final_attentions = Multiply()([x, attentions])
            final_attentions = tf.reduce_sum(final_attentions, axis=-1) #final_attentions.shape = (batch_sz, height, width, dim_cap)
            
            # Feature Extraction
            conv2 = tf.nn.conv2d(final_attentions, FeaExt_w, strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.bias_add(conv2, FeaExt_b)
            conv2 = tf.nn.relu(conv2)
            
            conv3 = tf.nn.conv2d(conv2, CapsAct_w, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.bias_add(conv3, CapsAct_b)
            conv3 = tf.expand_dims(conv3, axis=-1)
            return conv3
        
        outputs =  [func(Att_input, Att_w, FeaExt_w, CapsAct_w, FeaExt_b, CapsAct_b) for Att_input, Att_w, FeaExt_w, CapsAct_w, FeaExt_b, CapsAct_b in 
                   zip(Att_inputs, Att_ws, FeaExt_ws, CapsAct_ws, FeaExt_bs, CapsAct_bs)]
        outputs = tf.concat(outputs, axis=-1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return tuple([None, int(self.height/self.stride), int(self.width/self.stride), self.dim_capsule, self.n_channels])
    
    def get_config(self):
        config = {
            'n_channels': self.n_channels,
            'dim_capsule': self.dim_capsule
        }
        base_config = super(ConvCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class FullyConvCaps(layers.Layer):
    def __init__(self, n_channels, dim_capsule, kernel_regularizer=None, **kwargs):
        super(FullyConvCaps, self).__init__(**kwargs)

        self.n_channels = n_channels
        self.dim_capsule = dim_capsule
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.input_dim = input_shape[3]
        self.input_ch = input_shape[4]
        
        
        self.Att_W = self.add_weight(shape=[1, 1, self.dim_capsule, self.input_ch, self.input_ch*self.n_channels],
                                     initializer='glorot_uniform',  name='Att_W',trainable=True)
        self.ConvTrans_W = self.add_weight(shape =[3, 3, self.input_dim, self.input_ch*self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='ConvTrans_W',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.ConvTrans_B = self.add_weight(shape=[self.input_ch*self.dim_capsule*self.n_channels],
                                           initializer='zeros', name='ConvTrans_B',trainable=True)
        self.FeaExt_W = self.add_weight(shape =[self.height, self.width, self.dim_capsule, self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='FeaExt_W',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.FeaExt_B = self.add_weight(shape=[self.dim_capsule*self.n_channels], initializer='zeros', name='FeaExt_B',trainable=True)
        
        self.CapsAct_W = self.add_weight(shape =[1, 1, self.dim_capsule, self.dim_capsule*self.n_channels],
                                       initializer='glorot_uniform',  name='CapsAct_W',
                                       regularizer=self.kernel_regularizer, trainable=True)
        self.CapsAct_B = self.add_weight(shape=[self.dim_capsule*self.n_channels], initializer='zeros', name='CapsAct_B',trainable=True)
        
        
    def call(self, inputs):
        input_caps = tf.split(inputs, self.input_ch, axis=-1)
        ConvTrans_ws = tf.split(self.ConvTrans_W, self.input_ch, axis=-1)
        ConvTrans_bs = tf.split(self.ConvTrans_B, self.input_ch, axis=-1)
                
        # Convolutional Transform by 3x3 conv 
        conv1s = [tf.nn.conv2d(tf.squeeze(input_cap, axis=-1), ConvTrans_w, strides=[1, 1, 1, 1], padding='SAME')
                  for input_cap, ConvTrans_w in zip(input_caps, ConvTrans_ws)]
        conv1s = [tf.reshape(tf.nn.bias_add(conv1, ConvTrans_b), [-1, self.height, self.width, self.dim_capsule, self.n_channels, 1])
                  for conv1, ConvTrans_b in zip(conv1s, ConvTrans_bs)]
        conv1s = tf.concat(conv1s, axis=-1)
        conv1s = tf.transpose(conv1s, [0,1,2,3,5,4])
        
        # Att_inputs shape : (n_ch, batch_sz, h, w, dim_cap, input_ch, 1)
        Att_inputs = tf.split(conv1s, self.n_channels, axis=-1)
        Att_ws = tf.split(self.Att_W, self.n_channels, axis=-1)
        FeaExt_ws = tf.split(self.FeaExt_W, self.n_channels, axis=-1)
        FeaExt_bs = tf.split(self.FeaExt_B, self.n_channels, axis=-1)
        CapsAct_ws = tf.split(self.CapsAct_W, self.n_channels, axis=-1)
        CapsAct_bs = tf.split(self.CapsAct_B, self.n_channels, axis=-1)
        
            
        def func(conv1, Att_w, FeaExt_w, CapsAct_w, FeaExt_b, CapsAct_b) :
            x = tf.squeeze(conv1, axis=-1) #x.shape = (batch_sz, height, width, dim, input_ch)
            
            # Attention Routing
            # attentions shape =(batch_sz, height, width, 1, input_ch)
            attentions = tf.nn.conv3d(x, Att_w, strides=[1, 1, 1, 1, 1], padding='VALID')
            attentions = tf.nn.softmax(attentions, axis=-1)
            final_attentions = Multiply()([x, attentions])
            final_attentions = tf.reduce_sum(final_attentions, axis=-1) #final_attentions.shape = (batch_sz, height, width, dim)
            final_attentions = Dropout(rate=0.5)(final_attentions)
            
            # Feature Extraction
            conv2 = tf.nn.conv2d(final_attentions, FeaExt_w, strides=[1, 1, 1, 1], padding='VALID')
            conv2 = tf.nn.bias_add(conv2, FeaExt_b)
            conv2 = tf.nn.relu(conv2)
            
            conv3 = tf.nn.conv2d(conv2, CapsAct_w, strides=[1, 1, 1, 1], padding='SAME')
            conv3 = tf.nn.bias_add(conv3, CapsAct_b)
            return conv3
        
        outputs = [func(Att_input, Att_w, FeaExt_w, CapsAct_w, FeaExt_b, CapsAct_b) for Att_input, Att_w, FeaExt_w, CapsAct_w, FeaExt_b, CapsAct_b in 
                   zip(Att_inputs, Att_ws, FeaExt_ws, CapsAct_ws, FeaExt_bs, CapsAct_bs)]
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.reshape(outputs, [-1, self.dim_capsule, self.n_channels])
        outputs = tf.transpose(outputs, [0, 2, 1])
        return outputs
    
    def compute_output_shape(self, input_shape):
        return tuple([None, self.n_channels, self.dim_capsule])

    def get_config(self):
        config = {
            'n_channels': self.n_channels,
            'dim_capsule': self.dim_capsule
        }
        base_config = super(FullyConvCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Conv2d_bn(input_tensor, filters, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=None):
    x = Conv2D(filters, kernel_size = kernel_size, strides= strides, padding=padding, activation=None,
              kernel_regularizer=kernel_regularizer)(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(activation)(x)
    return x


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def build(self, input_shape):
        self.dim_capsule = input_shape[-1]
    
    def call(self, inputs, **kwargs):
        normalizing_tensor = tf.constant(np.sqrt(self.dim_capsule), dtype=tf.float32)
        output = K.sqrt(K.sum(K.square(inputs), -1)) / normalizing_tensor
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))



