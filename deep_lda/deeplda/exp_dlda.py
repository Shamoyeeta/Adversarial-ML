#!/usr/bin/env python
# author: matthias dorfer


# --- imports ---

from __future__ import print_function

import os
import pickle
import theano
import lasagne
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time as tm

from utils.data import load_mnist, load_cifar10, load_stl10
from utils.train_lda import fit, LDA, accuracy_score
from models.batch_iterators import batch_compute1

# root folder of experiments
# (parameters of trained model get dumped here)
EXP_ROOT = './model_params/'


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate DeepLDA models.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--data', help='select data set.')
    parser.add_argument('--predict', help='predict using DeepLDA model.')
    parser.add_argument('--train', help='train DeepLDA model.', action='store_true')
    parser.add_argument('--eval', help='evaluate DeepLDA model.', action='store_true')
    parser.add_argument('--fold', help='select train fold (only for stl10, (0-9), default=0).', type=int, default=0)
    parser.add_argument('--adv_sample', help='generate adversarial sample for the model.', action='store_true')
    args = parser.parse_args()

    # select model
    if str(args.model) == "mnist_dlda":
        from models import mnist_dlda as model
    elif str(args.model) == "cifar10_dlda":
        from models import cifar10_dlda as model
    elif str(args.model) == "stl10_dlda":
        from models import stl10_dlda as model
    else:
        pass

    print("\nLoading data ...")
    if str(args.data) == 'mnist':
        data = load_mnist()
    if str(args.data) == 'mnist_60k':
        data = load_mnist(k60=True)
    elif str(args.data) == 'cifar10':
        data = load_cifar10()
    elif str(args.data) == 'cifar10_50k':
        data = load_cifar10(k50=True)
    elif str(args.data) == 'stl10':
        data = load_stl10(fold=args.fold)
    else:
        pass

    # path to net dump
    exp_root = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = os.path.join(exp_root, 'params.pkl')

    # create folder for model parameters
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)

    print("\nCompiling network %s ..." % model.EXP_NAME)
    l_out, l_in = model.build_model()

    # -----------
    # train model
    # -----------
    if args.train:

        tr_batch_iter = model.train_batch_iterator()
        va_batch_iter = model.valid_batch_iterator()
        l_out, va_loss = fit(l_out, l_in, data, model.objective, model.Y_TENSOR_TYPE,
                             train_batch_iter=tr_batch_iter, valid_batch_iter=va_batch_iter,
                             r=model.r, num_epochs=100, patience=model.PATIENCE,
                             learn_rate=model.INI_LEARNING_RATE, update_learning_rate=model.update_learning_rate,
                             l_2=model.L2, compute_updates=model.compute_updates,
                             exp_name=model.EXP_NAME, out_path=exp_root, dump_file=dump_file)

    # --------------
    # evaluate model
    # --------------
    if args.eval:

        print("\nLoading model parameters from:")
        print(" (%s)" % dump_file)
        with open(dump_file, 'rb') as fp:
             u = pickle._Unpickler( fp )
             u.encoding = 'latin1'
             params = u.load()

        print("Setting model parameters ...")
        lasagne.layers.set_all_param_values(l_out, params)

        X_tr, y_tr = data['X_train'], data['y_train']
        X_va, y_va = data['X_valid'], data['y_valid']
        X_te, y_te = data['X_test'], data['y_test']

        # compile prediction function
        net_out = lasagne.layers.get_output(l_out, deterministic=True)
        compute_output = theano.function(inputs=[l_in.input_var], outputs=net_out)

        # compute network output
        print("\nComputing network output ...")
        net_output_tr = batch_compute1(X_tr, compute_output, model.BATCH_SIZE)
        net_output_va = batch_compute1(X_va, compute_output, model.BATCH_SIZE)
        net_output_te = batch_compute1(X_te, compute_output, model.BATCH_SIZE)

        # compute lda on net outputs
        print("\nComputing lda on network output ...")
        dlda = LDA(r=model.r, n_components=model.n_components, verbose=True, show=True)
        dlda.fit(net_output_tr, y_tr)

        # predict on test set
        print("\nComputing accuracies ...")
        y_pr = np.argmax(dlda.predict_proba(net_output_tr), axis=1)
        print("LDA Accuracy on train set: %.3f" % (100 * accuracy_score(y_tr, y_pr)))
        y_pr = np.argmax(dlda.predict_proba(net_output_va), axis=1)
        print("LDA Accuracy on valid set: %.3f" % (100 * accuracy_score(y_va, y_pr)))
        y_pr = np.argmax(dlda.predict_proba(net_output_te), axis=1)
        print("LDA Accuracy on test set:  %.3f" % (100 * accuracy_score(y_te, y_pr)))

        # project data to DeepLDA space
        XU_tr = dlda.transform(net_output_tr)
        XU_te = dlda.transform(net_output_te)

        # scatter plot of projection components
        colors = plt.cm.jet(np.arange(0.0, 1.0, 0.1))[:, 0:3]
        fig1 = plt.figure('DeepLDA-Feature-Scatter-Plot', facecolor='white')
        plt.clf()
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        plot_idx = 1
        for i in range(np.min([9, model.n_components])):
            for j in range(i + 1, np.min([9, model.n_components])):
                plt.subplot(6, 6, plot_idx)
                plot_idx += 1
                for l in np.unique(y_te):
                    idxs = y_te == l
                    plt.plot(XU_te[idxs, i], XU_te[idxs, j], 'o', color=colors[l], alpha=0.5)
                plt.axis('off')
                plt.axis('equal')
        fig1.savefig('scatter_plot.png')

        # plot histograms of features
        plt.figure('DeepLDA-Feature-Histograms', facecolor='white')
        plt.clf()
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
        for i in range(model.n_components):
            plt.subplot(5, 2, i + 1)

            F = XU_te[:, i]
            min_val, max_val = F.min(), F.max()

            for c in range(model.n_classes):
                F = XU_te[y_te == c, i]
                hist, bin_edges = np.histogram(F, range=(min_val, max_val), bins=100)
                plt.plot(bin_edges[1:], hist, '-', color=colors[c], linewidth=2)
            plt.axis('off')

        plt.show(block=True)

    # --------------------
    # Predict using model
    # --------------------
    if args.predict:

        print("\nLoading model parameters from:")
        print(" (%s)" % dump_file)
        with open(dump_file, 'rb') as fp:
            u = pickle._Unpickler(fp)
            u.encoding = 'latin1'
            params = u.load()

        print("Setting model parameters ...")
        lasagne.layers.set_all_param_values(l_out, params)

        # prepare data to pass

        adv_sample = args.predict
        print("Reading adversarial samples file from \n (", adv_sample, ")")
        with open(adv_sample, 'rb') as fp:
            adv = pickle.load(fp)
            x_adv = adv['x_adv']
            labels = adv['label']

        x_adv = x_adv.reshape(x_adv.shape[0], 1, 28, 28)
        #print(x_adv)

        # plot first 10 images
        num = 10
        images = x_adv[:num]
        num_row = 2
        num_col = 5
        # plot images
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num, 2.0 * num))
        for i in range(num):
            ax = axes[i // num_col][i % num_col]
            image = images[i]
            ax.imshow(image.reshape(28, 28), cmap='gray')
            ax.set_title('Label: {}'.format(labels[i]))
            plt.tight_layout(h_pad=20, w_pad=3)
        plt.show()

        # compile prediction function
        net_out = lasagne.layers.get_output(l_out, deterministic=True)
        result = theano.tensor.argmax(net_out, axis=1)
        compute1 = theano.function([l_in.input_var], net_out)
        compute = theano.function([l_in.input_var], result)

        print("Predicting output ...")
        # # compute results on batch
        net_output_1 = batch_compute1(x_adv, compute, model.BATCH_SIZE)
        print("\nComputing on x_adv...")
        net_output_2 = batch_compute1(x_adv, compute1, model.BATCH_SIZE)

        # compute lda on net outputs
        print("\nComputing lda on network output ...")
        dlda = LDA(r=model.r, n_components=model.n_components, verbose=True, show=True)

        start = tm.time()
        dlda.fit_from_file('./model_params/lda_obj_vals.pkl')
        print("elapsed time: ", tm.time() - start)

        # predict probabilities
        prob = dlda.predict_proba(net_output_2)
        # print(prob)
        # predict on adversarial set
        print("\nComputing accuracies ...")
        y_pr = np.argmax(dlda.predict_proba(net_output_2), axis=1)
        print('Finding most appropriate class labels - ',y_pr)
        print("LDA Accuracy on set: %.3f" % (100 * accuracy_score(labels, y_pr)))

    # -------------------
    # Train on LDA model
    # --------------------
    if args.adv_sample:
        print("\nLoading model parameters from:")
        print(" (%s)" % dump_file)
        with open(dump_file, 'rb') as fp:
            params = pickle.load(fp)

        print("Setting model parameters ...")
        lasagne.layers.set_all_param_values(l_out, params)

        # prepare data to pass

        X_tr, y_tr = data['X_train'], data['y_train']


