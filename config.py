#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys


FLAGS = tf.app.flags.FLAGS
#general variables
tf.app.flags.DEFINE_string('embedding_type','glove','can be: glove, word2vec-cbow, word2vec-SG, fasttext, BERT, BERT_Large, ELMo')
tf.app.flags.DEFINE_integer("year", 2016, "year data set [2014]")
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.07, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 200, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 19, 'max target length')

# traindata, testdata and embeddings, train path aangepast met ELMo
tf.app.flags.DEFINE_string("train_path_ont", "data/programGeneratedData/GloVetraindata"+str(FLAGS.year)+".txt", "train data path for ont")
tf.app.flags.DEFINE_string("test_path_ont", "data/programGeneratedData/GloVetestdata"+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("train_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) +str(FLAGS.embedding_dim)+'traindata'+str(FLAGS.year)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) + str(FLAGS.embedding_dim)+'testdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("embedding_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) + str(FLAGS.embedding_dim)+'embedding'+str(FLAGS.year)+".txt", "pre-trained glove vectors file path")
tf.app.flags.DEFINE_string("remaining_test_path_ELMo", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+"ELMo.txt", "only for printing")
tf.app.flags.DEFINE_string("remaining_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

# Data augmentation vars
tf.app.flags.DEFINE_string("EDA_type", "adjusted", "type of eda (original or adjusted)")
tf.app.flags.DEFINE_integer("EDA_deletion", 0, "number of deletion augmentations")
tf.app.flags.DEFINE_integer("EDA_replacement", 1, "number of replacement augmentations")
tf.app.flags.DEFINE_integer("EDA_insertion", 1, "number of insertion augmentations")
tf.app.flags.DEFINE_integer("EDA_swap", 3, "number of swap augmentations") # in adjusted mode, higher number means more swaps within the same category
tf.app.flags.DEFINE_float("EDA_pct", .2, "percentage of words affected by augmentation") # in adjusted mode EDA_swap not affected
# Note that each augmentation is a seperate sentence (so replacement = 1 means twice as much training data, so dont make the sum >10)
tf.app.flags.DEFINE_string("backtranslation_langs", "None", "languages to translate to, seperated by space, each space is an augmentation, write None if not using")
tf.app.flags.DEFINE_integer("use_word_mixup", 0, "boolean whether to use word mixup or not")
tf.app.flags.DEFINE_integer("use_sentence_mixup", 0, "boolean whether to use sentence mixup or not")
tf.app.flags.DEFINE_float("mixup_beta", 0.2, 'the alpha and beta values deciding the interpolation in mixup')
tf.app.flags.DEFINE_integer("mixup_on_augmentations", 0, "If mixup should be applied on augmentations or only on original data")
tf.app.flags.DEFINE_integer("original_multiplier", 1, "How many times the original data should be used in the training data")


# Data Augmentation paths
tf.app.flags.DEFINE_string('augmentation_file_path', "data/programGeneratedData/"+str(FLAGS.embedding_type)+'_augmented_data'+str(FLAGS.year)+".txt", "The file where augmented data is written to")
tf.app.flags.DEFINE_string('backtranslation_file', "data/programGeneratedData/translations.json", 'where backtranslation files are saved')


#svm traindata, svm testdata
tf.app.flags.DEFINE_string("train_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'trainsvmdata'+str(FLAGS.year)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'testsvmdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("remaining_svm_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingsvmtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#hyper traindata, hyper testdata
tf.app.flags.DEFINE_string("hyper_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+".txt", "hyper eval data path")

tf.app.flags.DEFINE_string("hyper_svm_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertrainsvmdata'+str(FLAGS.year)+".txt", "hyper train svm data path")
tf.app.flags.DEFINE_string("hyper_svm_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevalsvmdata'+str(FLAGS.year)+".txt", "hyper eval svm data path")

#external data sources
tf.app.flags.DEFINE_string("pretrain_file", "data/externalData/"+str(FLAGS.embedding_type)+"."+str(FLAGS.embedding_dim)+"d.txt", "pre-trained embedding vectors for non BERT and ELMo")

tf.app.flags.DEFINE_string("train_data", "data/externalData/restaurant_train_"+str(FLAGS.year)+".xml",
                    "train data path")
tf.app.flags.DEFINE_string("test_data", "data/externalData/restaurant_test_"+str(FLAGS.year)+".xml",
                    "test data path")

tf.app.flags.DEFINE_string('method', 'AE', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('prob_file', 'results.txt', 'prob')
tf.app.flags.DEFINE_string('results_file', 'results/data_augmentation_results2.json', 'files where results will be saved in json')
tf.app.flags.DEFINE_string('saver_file', 'prob1.txt', 'prob')


def print_config():
    #FLAGS._parse_flags()
    FLAGS(sys.argv)
    print('\nParameters:')
    for k, v in sorted(tf.app.flags.FLAGS.flag_values_dict().items()):
        print('{}={}'.format(k, v))


def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss


def acc_func(y, prob):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob


def train_func(loss, r, global_step, optimizer=None):
    if optimizer:
        return optimizer(learning_rate=r).minimize(loss, global_step=global_step)
    else:
        return tf.train.AdamOptimizer(learning_rate=r).minimize(loss, global_step=global_step)


def summary_func(loss, acc, test_loss, test_acc, _dir, title, sess):
    summary_loss = tf.summary.scalar('loss' + title, loss)
    summary_acc = tf.summary.scalar('acc' + title, acc)
    test_summary_loss = tf.summary.scalar('loss' + title, test_loss)
    test_summary_acc = tf.summary.scalar('acc' + title, test_acc)
    train_summary_op = tf.summary.merge([summary_loss, summary_acc])
    validate_summary_op = tf.summary.merge([summary_loss, summary_acc])
    test_summary_op = tf.summary.merge([test_summary_loss, test_summary_acc])
    train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
    test_summary_writer = tf.summary.FileWriter(_dir + '/test')
    validate_summary_writer = tf.summary.FileWriter(_dir + '/validate')
    return train_summary_op, test_summary_op, validate_summary_op, \
        train_summary_writer, test_summary_writer, validate_summary_writer


def saver_func(_dir):
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000)
    import os
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return saver
