# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #默认的显示等级,显示所有信息   1 -> 2
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU训练
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
"""

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim  #tensorflow辅助工具,用来简化代码
from tensorflow.python import debug as tf_debug

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')  # 参数的定义

# parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
# parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
# parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
# parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
# parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
# parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
# parser.add_argument('--input_height',              type=int,   help='input height', default=256)
# parser.add_argument('--input_width',               type=int,   help='input width', default=512)
# parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)  # 8->32
# parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
# parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
# parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
# parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
# parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
# parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
# parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
# parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
# parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1) # 1->0
# parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)  #8->16
# parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
# parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
# parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
# parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
# parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')


parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='train')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='/data/tion/kitti/')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default='/data/tion/my_data/all_npy_png.txt')
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=512)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-5)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1) # 4->1
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='/data/tion/tmp/my_loss_vgg_kitti_lr_batch4/')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')



args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path): # 计算文本文件行数  即训练数据量  一行存放一条训练数据
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines) # 返回行数 txt 文件中是照片的路径一共 29000 行 每一行都是一个双目图像 两张图的路径

def train(params):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'): # 创建graph

        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        # 根据步长改变学习率， 前3/5部分使用的是0.0001，后面每过1/5，将学习率除2
        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [args.learning_rate, args.learning_rate / 2, args.learning_rate / 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values) # 当走到一定步长时更改学习率

        opt_step = tf.train.AdamOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch

        # split for each gpu
        left_splits  = tf.split(left,  args.num_gpus, 0)
        right_splits = tf.split(right, args.num_gpus, 0)

        # ====================  coordinates
        left_coord = dataloader.left_coord_batch
        right_coord = dataloader.right_coord_batch

        # split for each gpu
        left_coord_splits = tf.split(left_coord, args.num_gpus, 0)
        right_coord_splits = tf.split(right_coord, args.num_gpus, 0)
        # ====================

        tower_grads  = []
        tower_losses = []
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                # with tf.device('/gpu:%d' % i):
                with tf.device('/gpu:1'):

                    model = MonodepthModel(params, args.mode, left_splits[i], right_splits[i], left_coord_splits[i], right_coord_splits[i], reuse_variables, i)
                    # model = MonodepthModel(params, args.mode, left_splits[i], right_splits[i],reuse_variables, i)

                    loss = model.total_loss
                    tower_losses.append(loss)

                    reuse_variables = True

                    grads = opt_step.compute_gradients(loss)

                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

        total_loss = tf.reduce_mean(tower_losses)

        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
        tf.summary.scalar('total_loss', total_loss, ['model_0'])
        summary_op = tf.summary.merge_all('model_0')

        # SESSION   通过tf.ConfigProto(allow_soft_placement=True) 来让Tensorflow自己选择可用设备
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # SAVER
        summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
        train_saver = tf.train.Saver()

        # COUNT PARAMS
        total_num_parameters = 0
        for variable in tf.trainable_variables():
            total_num_parameters += np.array(variable.get_shape().as_list()).prod()
        print("number of trainable parameters: {}".format(total_num_parameters))

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)


        # LOAD CHECKPOINT IF SET
        if args.checkpoint_path != '':
            train_saver.restore(sess, args.checkpoint_path.split(".")[0])

            if args.retrain:
                sess.run(global_step.assign(0))

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            before_op_time = time.time()
            _, loss_value, disp_gradient, lr_loss, image_loss, coord_image_loss, coord_lr_loss, coord_smooth_loss = sess.run(
                [apply_gradient_op, total_loss, model.disp_gradient_loss, model.lr_loss,model.image_loss,
                 model.coord_image_loss, model.coord_lr_loss, model.coord_smoothness_loss
                 ])
            # _, loss_value= sess.run([apply_gradient_op, total_loss])
            duration = time.time() - before_op_time

            if step and step % 100 == 0:  # 100 -> 200
                # ============================= test
                print('--------------------------------------')
                print('disp_gradient    lr_loss   image_loss    sum')
                print('%.8f | %.8f | %.8f | %.8f' % (
                    disp_gradient * 0.1, lr_loss, image_loss, (disp_gradient * 0.1 + lr_loss + image_loss)))
                print('%.8f | %.8f | %.8f | %.8f' % (
                    coord_smooth_loss * 0.1, coord_lr_loss, coord_image_loss,
                    (coord_smooth_loss * 0.1 + coord_lr_loss + coord_image_loss)))
                # =============================
                examples_per_sec = params.batch_size / duration
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / step - 1.0) * time_sofar
                print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step and step % 10000 == 0:
                train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

        train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

def test(params):
    """Test function."""

    dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
    left  = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = MonodepthModel(params, args.mode, left, right)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    if args.checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
    else:
        restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    num_test_samples = count_text_lines(args.filenames_file)

    print('now testing {} files'.format(num_test_samples))
    disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(num_test_samples):
        disp = sess.run(model.disp_left_est[0])
        disparities[step] = disp[0].squeeze()
        disparities_pp[step] = post_process_disparity(disp.squeeze())

    print('done.')

    print('writing disparities.')
    if args.output_directory == '':
        output_directory = os.path.dirname(args.checkpoint_path)
    else:
        output_directory = args.output_directory
    np.save(output_directory + '/disparities.npy',    disparities)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp)

    print('done.')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary)

    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
