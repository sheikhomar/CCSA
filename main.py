import os
import random
import sys

import numpy   as np
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.utils import np_utils


def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def Create_Pairs(domain_adaptation_task, repetition, sample_per_class):
    UM = domain_adaptation_task
    cc = repetition
    SpC = sample_per_class

    if UM != 'MNIST_to_USPS':
        if UM != 'USPS_to_MNIST':
            raise Exception('domain_adaptation_task should be either MNIST_to_USPS or USPS_to_MNIST')

    if cc < 0 or cc > 10:
        raise Exception('number of repetition should be between 0 and 9.')

    if SpC < 1 or SpC > 7:
        raise Exception('number of sample_per_class should be between 1 and 7.')

    print('Creating pairs for repetition: ' + str(cc) + ' and sample_per_class: ' + str(sample_per_class))

    X_train_target = np.load(
        './row_data/' + UM + '_X_train_target_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')
    y_train_target = np.load(
        './row_data/' + UM + '_y_train_target_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')

    X_train_source = np.load(
        './row_data/' + UM + '_X_train_source_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')
    y_train_source = np.load(
        './row_data/' + UM + '_y_train_source_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')

    Training_P = []
    Training_N = []

    for trs in range(len(y_train_source)):
        for trt in range(len(y_train_target)):
            if y_train_source[trs] == y_train_target[trt]:
                Training_P.append([trs, trt])
            else:
                Training_N.append([trs, trt])

    random.shuffle(Training_N)
    Training = Training_P + Training_N[:3 * len(Training_P)]
    random.shuffle(Training)

    X1 = np.zeros([len(Training), 16, 16], dtype='float32')
    X2 = np.zeros([len(Training), 16, 16], dtype='float32')

    y1 = np.zeros([len(Training)])
    y2 = np.zeros([len(Training)])
    yc = np.zeros([len(Training)])

    for i in range(len(Training)):
        in1, in2 = Training[i]
        X1[i, :, :] = X_train_source[in1, :, :]
        X2[i, :, :] = X_train_target[in2, :, :]

        y1[i] = y_train_source[in1]
        y2[i] = y_train_target[in2]
        if y_train_source[in1] == y_train_target[in2]:
            yc[i] = 1

    if not os.path.exists('./pairs'):
        os.makedirs('./pairs')

    np.save('./pairs/' + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X1)
    np.save('./pairs/' + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', X2)

    np.save('./pairs/' + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y1)
    np.save('./pairs/' + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', y2)
    np.save('./pairs/' + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy', yc)


def Create_Model():
    img_rows, img_cols = 16, 16
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dense(84))
    model.add(Activation('relu'))
    return model


def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def training_the_model(model, domain_adaptation_task, repetition, sample_per_class):
    nb_classes = 10
    UM = domain_adaptation_task
    cc = repetition
    SpC = sample_per_class

    if UM != 'MNIST_to_USPS':
        if UM != 'USPS_to_MNIST':
            raise Exception('domain_adaptation_task should be either MNIST_to_USPS or USPS_to_MNIST')

    if cc < 0 or cc > 10:
        raise Exception('number of repetition should be between 0 and 9.')

    if SpC < 1 or SpC > 7:
        raise Exception('number of sample_per_class should be between 1 and 7.')

    epoch = 80  # Epoch number
    batch_size = 256

    X_test = np.load(
        './row_data/' + UM + '_X_test_target_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')
    y_test = np.load(
        './row_data/' + UM + '_y_test_target_repetition_' + str(cc) + '_sample_per_class_' + str(SpC) + '.npy')
    X_test = X_test.reshape(X_test.shape[0], 16, 16, 1)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    X1 = np.load('./pairs/' + UM + '_X1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    X2 = np.load('./pairs/' + UM + '_X2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')

    X1 = X1.reshape(X1.shape[0], 16, 16, 1)
    X2 = X2.reshape(X2.shape[0], 16, 16, 1)

    y1 = np.load('./pairs/' + UM + '_y1_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    y2 = np.load('./pairs/' + UM + '_y2_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')
    yc = np.load('./pairs/' + UM + '_yc_count_' + str(cc) + '_SpC_' + str(SpC) + '.npy')

    y1 = np_utils.to_categorical(y1, nb_classes)
    y2 = np_utils.to_categorical(y2, nb_classes)

    print('Training the model - Epoch ' + str(epoch))
    nn = batch_size
    best_Acc = 0
    for e in range(epoch):
        if e % 10 == 0:
            printn(str(e) + '->')
        for i in range(int(len(y2) / nn)):
            loss = model.train_on_batch([X1[i * nn:(i + 1) * nn, :, :, :], X2[i * nn:(i + 1) * nn, :, :, :]],
                                        [y1[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])
            loss = model.train_on_batch([X2[i * nn:(i + 1) * nn, :, :, :], X1[i * nn:(i + 1) * nn, :, :, :]],
                                        [y2[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])

        Out = model.predict([X_test, X_test])
        Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_test, axis=1)
        Acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)

        if best_Acc < Acc:
            best_Acc = Acc
    print(str(e))
    return best_Acc


def main():
    # let's assume MNIST->USPS task.
    domain_adaptation_task = 'MNIST_to_USPS'  # USPS_to_MNIST is also another option.

    # let's run the experiments when 1 target sample per calss is available in training.
    # you can run the experiments for sample_per_class=1, ... , 7.
    sample_per_class = 1

    # Running the experiments for repetition 5. In the paper we reported the average acuracy.
    # We run the experiments for repetition=0,...,9 and take the average
    repetition = 2

    # Creating embedding function. This corresponds to the function g in the paper.
    # You may need to change the network parameters.
    model1 = Create_Model()

    # size of digits 16*16
    img_rows, img_cols = 16, 16
    input_shape = (img_rows, img_cols, 1)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # number of classes for digits classification
    nb_classes = 10

    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = .25

    # Having two streams. One for source and one for target.
    processed_a = model1(input_a)
    processed_b = model1(input_b)

    # Creating the prediction function. This corresponds to h in the paper.
    out1 = Dropout(0.5)(processed_a)
    out1 = Dense(nb_classes)(out1)
    out1 = Activation('softmax', name='classification')(out1)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='CSA')([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
    model.compile(loss={'classification': 'categorical_crossentropy', 'CSA': contrastive_loss},
                  optimizer='adadelta',
                  loss_weights={'classification': 1 - alpha, 'CSA': alpha})

    print('Domain Adaptation Task: ' + domain_adaptation_task)
    # let's create the positive and negative pairs using row data.
    # pairs will be saved in ./pairs directory
    sample_per_class = 1
    for repetition in range(10):
        Create_Pairs(domain_adaptation_task, repetition, sample_per_class)
        Acc = training_the_model(model, domain_adaptation_task, repetition, sample_per_class)

        print(
            'Best accuracy for {} target sample per class and repetition {} is {}.'.format(sample_per_class, repetition,
                                                                                           Acc))


if __name__ == '__main__':
    main()
