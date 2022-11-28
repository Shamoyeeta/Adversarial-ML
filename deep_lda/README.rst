Deep Linear Discriminant Analysis (DeepLDA)
===========================================

This repository contains code that has been adapted from the original DeepLDA model reported in the ICLR 2016 paper
`Deep Linear Discriminant Analysis <http://arxiv.org/abs/1511.04707>`_
by Matthias Dorfer, Rainer Kelz and Gerhard Widmer from the `Department of Computational Perception <http://www.cp.jku.at/>`_ at `JKU Linz <http://www.jku.at/>`_.

Requirements
------------

The implementation is based on `Theano <https://github.com/Theano/Theano>`_
and the neural networks library `Lasagne <https://github.com/Lasagne/Lasagne>`_.
For installing Theano and Lasagne please follow the installation instruction on the respective github pages.

You will also need: matplotlib, numpy and scipy


For training the models just run the following commands:

MNIST: the model should train up to a validation accuracy of around 99.7%.::

    python exp_dlda.py --model mnist_dlda --data mnist --train


For evaluating the trained models run the following commands.
    
    python exp_dlda.py --model mnist_dlda --data mnist --eval
    
The script will report:

* The accuracies on train, validation and test set
* Report the magnitudes of the individual eigenvalues after solving the general (Deep)LDA eigenvalue problem
* Produce some plots visualizing the structure of the latent representation produced by the model 

For checking accuracy of the trained model on adversarial example (saved on a local file images.pkl at same level as exp_dlda) run:
    
    python exp_dlda.py --model mnist_dlda --data mnist --predict images.pkl

