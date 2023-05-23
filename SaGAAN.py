import os, time, itertools, imageio, pickle, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mmd import mix_rbf_mmd2
import scipy.io
import pickle
#from scipy.interpolate import Bspline

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# G(z)

def generator5( x, y_label, isTrain=True, reuse=False):   #x=[None,1,50]  y_label=[None,1,9]
    with tf.variable_scope('generator', reuse=reuse):

        # initializer
        # initializer

        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        # concat layer
        cat1 = tf.concat([x, y_label], 2)   #[?,1,59]
        l = tf.shape(cat1)[0]

        cat1 = tf.reshape(cat1, [l,1,1,59])

        #cat1 = x
        # 1st hidden layer
        deconv1 = tf.layers.conv2d_transpose(cat1, 128, [1, 13], strides=(1, 1), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
        #lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)
        lrelu1 = tf.nn.relu(deconv1)
        #[?,1,13,128]
        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 256, [1,2], strides=(1, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        #lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)
        lrelu2 = tf.nn.relu(deconv2)
        #[?,1,26,256]
        # 3rd hidden layer
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 64, [1,2], strides=(1, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        #lrelu3 = lrelu(tf.layers.batch_normalization(deconv3, training=isTrain), 0.2)
        lrelu3 = tf.nn.relu(deconv3)
        # [?,1,52,128]

        # 4rd hidden layer
        deconv4 = tf.layers.conv2d_transpose(lrelu3, 1, [1, 2], strides=(1, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        #lrelu4 = lrelu(tf.layers.batch_normalization(deconv4, training=isTrain), 0.2)
        lrelu4 = tf.nn.relu(deconv4)

        #[?,1,104,1]
        lrelu4=tf.reshape(lrelu4,[l,1,1,104])
        #attention
        proj_query = tf.layers.conv2d_transpose(lrelu4, 1, [1, 2], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        proj_key = tf.layers.conv2d_transpose(lrelu4, 1, [1, 2], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        value = tf.transpose(proj_query, [0, 1, 3,2])
        enery = tf.matmul(value, proj_key)
        attention = tf.nn.softmax(enery)
        proj_value = tf.layers.conv2d_transpose(lrelu4, 1, [1, 2], strides=(1, 1), padding='same', kernel_initializer=w_init, bias_initializer=b_init)

        G = tf.matmul(proj_value, attention)
        G_a=G*gamma+lrelu4
        o = tf.reshape(G_a, [l, 104, 1])
        return o

def discriminator2(x, y_fill, isTrain=True,
                       reuse=False):  # G_z=[500,104,1]    y_fill=[None, img_size,9]  img_size=104
    with tf.variable_scope('discriminator', reuse=reuse):
        out = []
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        # w_init = tf.xavier_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, y_fill], 2)  # [?,104,10]

        # cat1 = x
         # 1st hidden layer
        conv1 = tf.layers.conv1d(cat1, 50, 10, strides=2, padding='VALID', kernel_initializer=w_init,
                                     bias_initializer=b_init)
        # lrelu1 = lrelu(conv1, 0.2)
        lrelu1 = tf.nn.relu(conv1)  # [?,47,50]
        out.append(lrelu1)
        # 2nd hidden layer
        conv2 = tf.layers.conv1d(lrelu1, 100, 10, strides=2, padding='VALID', kernel_initializer=w_init,
                                     bias_initializer=b_init)
        # lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        lrelu2 = tf.nn.relu(conv2)  # [?,19,100]
        out.append(lrelu2)
        # 3rd hidden layer
        conv3 = tf.layers.conv1d(lrelu2, 200, 10, strides=2, padding='VALID', kernel_initializer=w_init,
                                     bias_initializer=b_init)
        lrelu3 = tf.nn.relu(conv3)  # [?,5,200]
        out.append(lrelu3)
        # output layer
        conv4 = tf.layers.conv1d(lrelu3, 50, 3, strides=1, padding='valid', kernel_initializer=w_init,
                                     bias_initializer=b_init)
        lrelu4 = tf.nn.relu(conv4)  # [?,3,50]
        out.append(lrelu4)

        conv5 = tf.layers.conv1d(lrelu4, 9, 4, strides=1, padding='valid', kernel_initializer=w_init,
                                     bias_initializer=b_init)
        lrelu5 = tf.nn.softmax(conv5)  # [?,1,9]

        conv6 = tf.layers.conv1d(lrelu4, 1, 4, strides=1, padding='valid', kernel_initializer=w_init,
                                     bias_initializer=b_init)
        # [?,1,1]


        o = tf.nn.relu(conv6)

        test_logits = lrelu5
        return out, o, conv6, test_logits

# preprocess
img_size = 104
onehot = np.eye(9)
temp_z_ = np.random.normal(0, 1, (81, 1, 50))
fixed_z_ = temp_z_
fixed_y_ = np.zeros((9, 1))
for i in range(8):
    # fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
    temp = np.ones((9, 1)) + i
    fixed_y_ = np.concatenate([fixed_y_, temp], 0)
# temp_x_=np.random.normal(0,1,(81,103,1))
fixed_y_ = onehot[fixed_y_.astype(np.int32)].reshape((81, 1, 9))


def show_result(num_epoch, show=False, save=False, path='result.png'):
    test_images = sess.run(G_z, {z: temp_z_, y_label: fixed_y_, isTrain: False})  # replace fixed_z_

    size_figure_grid = 9
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(100, 100))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(True)
        ax[i, j].get_yaxis().set_visible(True)
        ax[i, j].set_ylim([0, 1])

    for k in range(9 * 9):
        i = k // 9
        j = k % 9
        ax[i, j].cla()
        x1 = list(range(104))
        # y_smooth = spline(x, test_images[k], x)
        # ax[i, j].plot(np.reshape(test_images[k], (img_size, 1)))
        ax[i, j].plot(x1, test_images[k], 'black', linewidth=1)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


temp_c_ = np.random.normal(0, 1, (4273, 1, 50))
y1 = scipy.io.loadmat('with_cluster/trY_withCluster.mat')['label_all']
fixed_c_ = y1.reshape((4273, 1, 9))


def save_generator_results(epoch, root):
    test_images = sess.run(G_z, {z: temp_c_, y_label: fixed_c_, isTrain: False})  # replace fixed_z_
    namepre = root + str(epoch + 1) + 'preimage' + '.mat'

    scipy.io.savemat(namepre, {'specisl': test_images})


def save_generated_samples(temp_z, temp_y):
    test_images = sess.run(G_z, {z: temp_z, y_label: temp_y, isTrain: False})
    f1 = open('generated_samples_data', 'wb')
    f2 = open('generated_samples_label', 'wb')
    pickle.dump(test_images, f1)
    pickle.dump(temp_y, f2)
    f1.close()
    f2.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_size = 1000
dlr = 0.001  # 0.0001 for without clustering
glr = 0.0001

train_epoch = 1000
global_step = tf.Variable(0, trainable=False)
train_mode = 1
z_dim = 50
log_device_placement = True
# ******************************************************************************
# load training samples from pkl files

trX = scipy.io.loadmat('with_cluster/Train.mat')['spec_all']
trY = scipy.io.loadmat('with_cluster/trY_withCluster.mat')['label_all']
# teX = scipy.io.loadmat('spec_all_test.mat')['spec_all']
# teY = scipy.io.loadmat('teY.mat')['label_all']
train_set = trX
train_label = trY



# sortmax transform onehot
def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b


# lr = tf.train.exponential_decay(0.00001, global_step, 500, 0.95, staircase=True)
# load MNIST
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

with tf.device('/gpu:0'):
    # variables : input
    x = tf.placeholder(tf.float32, shape=(None, img_size, 1))  # img_size=103  batch_size=500
    z = tf.placeholder(tf.float32, shape=(None, 1, z_dim))  # z_dim=50
    y_label = tf.placeholder(tf.float32, shape=(None, 1, 9))
    y_fill = tf.placeholder(tf.float32, shape=(None, img_size, 9))
    isTrain = tf.placeholder(dtype=tf.bool)

    # networks : generator
    G_z = generator5(z, y_label, isTrain)
    # G=attention('attention',G_z)

    # networks : discriminator
    layer_out_r, D_real, D_real_logits, D_pre_labels = discriminator2(x, y_fill, isTrain)
    layer_out_f, D_fake, D_fake_logits, _ = discriminator2(G_z, y_fill, isTrain, reuse=True)

    # loss for each network
    if train_mode==1:
        #MMD
        image=tf.reshape(x, [batch_size, -1])
        G=tf.reshape(G_z,[batch_size, -1])
        kernel_loss = mix_rbf_mmd2(G, image)
        ada_loss = tf.sqrt(kernel_loss)
    else:
        #adaptation
        f_match = tf.constant(0., dtype=tf.float32)
        for i in range(4):
            f_match += tf.reduce_mean(tf.multiply(layer_out_f[i] - layer_out_r[i], layer_out_f[i] - layer_out_r[i]))
        ada_loss=f_match
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1])))
    D_loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_pre_labels, labels=y_label))
    D_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1])))
    D_loss = D_loss_real + D_loss_fake + D_loss_dis + ada_loss

    G_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1])))

    # trainable variables for each network
    T_vars = tf.trainable_variables()
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]

    # test

    # optimizer for each network
with tf.device('/cpu:0'):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optim = tf.train.AdamOptimizer(dlr, beta1=0.5)

        D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)

        # D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
        G_optim = tf.train.AdamOptimizer(glr, beta1=0.5).minimize(G_loss, var_list=G_vars)

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True
# open session and initialize all variables
# sess = tf.InteractiveSession()
root = 'MNIST_cDCGAN_results/'
model = 'MNIST_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_spec'):
    os.mkdir(root + 'Fixed_spec')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['D_dis_losses'] = []

train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # training-loop
    np.random.seed(int(time.time()))
    print('training start!')
    start_time = time.time()
    if train_mode == 1:
        for epoch in range(train_epoch):
            G_losses = []
            D_losses = []
            G_dis = []
            G_dia = []

            epoch_start_time = time.time()
            shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0])
            shuffled_set = train_set[shuffle_idxs]
            shuffled_label = train_label[shuffle_idxs]
            for iter in range(shuffled_set.shape[0] // batch_size):  # 4273//500
                # update discriminator
                x_ = shuffled_set[iter * batch_size:(iter + 1) * batch_size].reshape([batch_size, 104, 1])
                y_label_ = shuffled_label[iter * batch_size:(iter + 1) * batch_size].reshape([batch_size, 1, 9])
                y_fill = y_label_ * np.ones([batch_size, img_size, 9])
                z_ = np.random.normal(0, 1, (batch_size, 1, z_dim))
                loss_d_, _ = sess.run([D_loss, D_optim],
                                      {x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, isTrain: True})

                # update generator
                z_ = np.random.normal(0, 1, (batch_size, 1, z_dim))
                y_ = np.random.randint(0, 8, (batch_size, 1))
                y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 9])
                y_fill_ = y_label_ * np.ones([batch_size, img_size, 9])
                loss_g_, _ = sess.run([G_loss, G_optim],
                                      {z: z_, x: x_, y_fill: y_fill_, y_label: y_label_, isTrain: True})

                errD_fake = D_loss_fake.eval({z: z_, x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
                errD_real = D_loss_real.eval({x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
                errD_dis = D_loss_dis.eval({x: x_, y_label: y_label_, y_fill: y_fill, isTrain: False})

                errG = G_loss.eval({x: x_, z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})

                errD_total = errD_fake + errD_real + errD_dis

                D_losses.append(errD_total)
                G_losses.append(errG)
                G_dis.append(errD_dis)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f, loss_dis: %.3f' % (
            (epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses), np.mean(G_dis)))
            fixed_p = root + 'Fixed_spec/' + model + str(epoch + 1) + '.png'
            abc = root + 'resultData/'
            show_result((epoch + 1), save=True, path=fixed_p)
            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))
            train_hist['D_dis_losses'].append(np.mean(G_dis))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
            if (epoch + 1) == train_epoch:
                save_generator_results(epoch, abc)
                print('model saving..............')
                path = os.getcwd() + '/model_saved'
                save_path = os.path.join(path, "model.ckpt")
                saver = tf.train.Saver()
                saver.save(sess, save_path=save_path)
                print('model saved...............')

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (
        np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
        print("Training finish!... save training results")
        with open(root + model + 'train_hist.pkl', 'wb') as f:
            pickle.dump(train_hist, f)

        show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
    sess.close()
