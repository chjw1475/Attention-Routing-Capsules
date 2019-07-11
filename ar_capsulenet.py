import numpy as np
import os
import time
from keras import layers, models, optimizers, callbacks, regularizers, activations, layers, initializers
from keras import backend as K
from PIL import Image
from keras.layers import Dense, Input, Activation, Reshape, Add
import tensorflow as tf
from ar_capsulelayers import *

K.set_image_data_format('channels_last')

def AR_CapsNet(input_shape, args):
    dim_caps = int(args.dimcaps)
    layernum = int(args.layernum)
    print('layer num : ', layernum)
    print('dim_caps : ', dim_caps)
    
    kernel_regularizer=regularizers.l2(0)
    input_layer = Input(shape=input_shape)
    conv1 = Conv2d_bn(input_tensor = input_layer, filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_regularizer=kernel_regularizer)
    conv1 = Conv2d_bn(input_tensor = conv1, filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_regularizer=kernel_regularizer)
    
    ## Primary Capsules
    primarycaps = PrimaryCap(n_channels=8, dim_capsule=16, kernel_regularizer=kernel_regularizer)(conv1)
    primarycaps = Activation('tanh')(primarycaps)
    print('primary caps shape : ', primarycaps.shape)
        
    ## Convolutional Capsules
    if layernum == 0:
        out = primarycaps
    elif layernum == 1:
        ConvCaps1 = ConvCaps(n_channels=8, dim_capsule=dim_caps, decrease_resolution = True, kernel_regularizer=kernel_regularizer)(primarycaps)
        ConvCaps1 = Activation('tanh')(ConvCaps1)
        print('ConvCaps1 shape : ', ConvCaps1.shape)
        out = ConvCaps1
        
    elif layernum == 2:
        ConvCaps1 = ConvCaps(n_channels=8, dim_capsule=dim_caps, decrease_resolution = True, kernel_regularizer=kernel_regularizer)(primarycaps)
        ConvCaps1 = Activation('tanh')(ConvCaps1)
        print('ConvCaps1 shape : ', ConvCaps1.shape)
        
        ConvCaps2 = ConvCaps(n_channels=8, dim_capsule=dim_caps, decrease_resolution = False, kernel_regularizer=kernel_regularizer)(ConvCaps1)
        ConvCaps2 = Activation('tanh')(Add()([ConvCaps11 , ConvCaps2]))
        print('ConvCaps2 shape : ', ConvCaps2.shape)
        out = ConvCaps2
        
    elif layernum == 3:
        ConvCaps1 = ConvCaps(n_channels=8, dim_capsule=dim_caps, decrease_resolution = True, kernel_regularizer=kernel_regularizer)(primarycaps)
        ConvCaps1 = Activation('tanh')(ConvCaps1)
        print('ConvCaps1 shape : ', ConvCaps1.shape)
        
        ConvCaps2 = ConvCaps(n_channels=8, dim_capsule=dim_caps, decrease_resolution = False, kernel_regularizer=kernel_regularizer)(ConvCaps1)
        ConvCaps2 = Activation('tanh')(Add()([ConvCaps1 , ConvCaps2]))
        print('ConvCaps2 shape : ', ConvCaps2.shape)
        
        ConvCaps3 = ConvCaps(n_channels=8, dim_capsule=dim_caps, decrease_resolution = False, kernel_regularizer=kernel_regularizer)(ConvCaps2)
        ConvCaps3 = Activation('tanh')(Add()([ConvCaps2 , ConvCaps3]))
        print('ConvCaps3 shape : ', ConvCaps3.shape)
        out = ConvCaps3
        
    ## Fully Convolutional Capsules
    output_dim_capsule = dim_caps 
    outputs = FullyConvCaps(n_channels=10, dim_capsule=output_dim_capsule, kernel_regularizer=kernel_regularizer)(out)
    outputs = Activation('tanh')(outputs)
    print('Final Routing caps shape : ', outputs.shape)
    
    ## Length Capsules
    real_outputs = Length()(outputs)
    print('Length shape : ', real_outputs.shape)

    n_class=10
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([outputs, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(outputs)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(512, activation='relu', input_dim=output_dim_capsule*n_class, kernel_regularizer=kernel_regularizer))
    decoder.add(Dense(512, activation='relu', kernel_regularizer=kernel_regularizer))
    decoder.add(Dense(np.prod(input_shape), activation='sigmoid', kernel_regularizer=kernel_regularizer))
    decoder.add(Reshape(target_shape=input_shape, name='out_recon'))

    train_model = models.Model([input_layer, y], [real_outputs, decoder(masked_by_y)])
    eval_model = models.Model(input_layer, [real_outputs, decoder(masked)])
    perturb_input_model = models.Model([input_layer, y], decoder(masked_by_y))

    # manipulate model
    noise = layers.Input(shape=(n_class, output_dim_capsule))
    noised_outputs = layers.Add()([outputs, noise])
    masked_noised_y = Mask()([noised_outputs, y])
    manipulate_model = models.Model([input_layer, y, noise], [outputs, decoder(masked_noised_y)])
    return train_model
    
def train(train_model, x_train, y_train, args):
    valid_ratio = 0.1*int(args.validratio)
    print('valid_ratio', valid_ratio)
    
    ##########################################################################################################
    # Training without data augmentation.
    # callbacks
    class TimeHistory(callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)
    
    time_callback = TimeHistory()
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_length_1_acc',
                                           save_best_only=True, save_weights_only=False, verbose=1)
    def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 50:
            lrate = 0.0005
        elif epoch > 200:
            lrate = 0.0001
        return lrate

    initial_lr = 0.005
    RMSprop = optimizers.RMSprop(lr=initial_lr, rho=0.9, epsilon=1e-08, decay=1e-4)
    train_model.compile(optimizer=RMSprop,
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., 0.3],
                      metrics=['acc'])
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: initial_lr * (0.99 ** epoch))
    
    if args.augment == 'False':
        # Training without data augmentation:
        hist = train_model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs, 
                               validation_split=valid_ratio,
                               callbacks=[callbacks.LearningRateScheduler(lr_schedule), log, tb, checkpoint, time_callback])
        
    elif args.augment == 'True':
        from keras.preprocessing.image import ImageDataGenerator
        shift_fraction = args.shift_fraction
        if args.dataset == 'mnist':
            flip = False
        elif args.dataset == 'cifar10':
            flip = True
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=shift_fraction,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=shift_fraction,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=flip,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None)

        def train_generator(x, y, batch_size):
            train_datagen = datagen  

            generator = train_datagen.flow(x, y, batch_size=batch_size)
            while 1:
                x_batch, y_batch = generator.next()
                yield ([x_batch, y_batch], [y_batch, x_batch])
        
        from sklearn.model_selection import train_test_split
        x_train_, x_val_, y_train_, y_val_ = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=42)


        hist = train_model.fit_generator(generator=train_generator(x_train_, y_train_, args.batch_size),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            verbose=1,
                            validation_data = [[x_val_, y_val_], [y_val_, x_val_]],
                            callbacks=[callbacks.LearningRateScheduler(lr_schedule), log, tb, checkpoint, time_callback])

    train_model.save(args.save_dir + '/trained_model.h5')
    print(time_callback.times)
    # writedata.py
    f = open(args.save_dir+'/time.txt', 'w')
    for t in time_callback.times:
        f.write(str(t)+'\n')
    f.close()


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    from keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
#     return (x_train[:500,...], y_train[:500,...]), (x_test, y_test)
    return (x_train, y_train), (x_test, y_test)

def cifar10():
    # the data, shuffled and split between train and test sets
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
#     return (x_train[:500,...], y_train[:500,...]), (x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import argparse
    
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="AR Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
#    parser.add_argument('-t', '--testing', action='store_true',
#                        help="Test the trained model on testing dataset")
#    parser.add_argument('-w', '--weights', default=None,
#                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--augment', default='False',help="augmentation")
    parser.add_argument('--gpu', default="0",help="gpu")
    parser.add_argument('--dataset', default="mnist", help="dataset")
    parser.add_argument('--layernum', default="0", help="layernum")
    parser.add_argument('--dimcaps', default="16", help="dimcaps")
    parser.add_argument('--validratio', default="1", help="validratio")
    #parser.add_argument('--log_dir', default='./result')
    

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist()
    elif args.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10()
    
    model = AR_CapsNet(x_train[0].shape, args)
    
    print(model.summary())

    train(model, x_train, y_train, args)
        
        
        