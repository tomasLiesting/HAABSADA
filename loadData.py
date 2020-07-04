from dataReader2016 import read_data_2016
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
from tqdm import tqdm
import io


def loadDataAndEmbeddings(config, loadData, augment_data):
    FLAGS = config

    if loadData == True:
        source_count, target_count = [], []
        source_word2idx, target_phrase2idx = {}, {}

        print('reading training data...')
        train_data, ct = read_data_2016(FLAGS.train_data,
                                        source_count,
                                        source_word2idx,
                                        target_count,
                                        target_phrase2idx,
                                        FLAGS.train_path,
                                        augment_data,
                                        FLAGS.augmentation_file_path)
        print('reading test data...')
        test_data, _ = read_data_2016(FLAGS.test_data,
                                      source_count,
                                      source_word2idx,
                                      target_count,
                                      target_phrase2idx,
                                      FLAGS.test_path,
                                      False,
                                      None)

        print('creating embeddings...')
        wt = np.random.normal(0, 0.05, [len(source_word2idx), 300])
        word_embed = {}
        count = 0.0
        with open(FLAGS.pretrain_file, 'r', encoding="utf8") as f:
            for line in tqdm(f):
                content = line.strip().split()
                if content[0] in source_word2idx:
                    wt[source_word2idx[content[0]]] = np.array(list(map(float, content[1:])))
                    count += 1

        print('finished embedding context vectors...')

        # print data to txt file
        outF = io.open(FLAGS.embedding_path, "w", encoding='utf-8')
        for i, word in enumerate(source_word2idx):
            outF.write(word)
            outF.write(" ")
            outF.write(' '.join(str(w) for w in wt[i]))
            outF.write("\n")
        outF.close()
        print('wrote the embedding vectors to file')
        print((len(source_word2idx) - count) / len(source_word2idx) * 100)

        return len(train_data[0]), len(test_data[0]), train_data[4], test_data[
            4], ct  # train_size, test_size, train_polarity_vector, test_polarity_vector

    else:
        # get statistic properties from txt file
        train_size, train_polarity_vector = getStatsFromFile(FLAGS.train_path)
        test_size, test_polarity_vector = getStatsFromFile(FLAGS.test_path)

        return train_size, test_size, train_polarity_vector, test_polarity_vector


def loadAverageSentence(config, sentences, pre_trained_context):
    FLAGS = config
    wt = np.zeros((len(sentences), FLAGS.edim))
    for id, s in enumerate(sentences):
        for i in range(len(s)):
            wt[id] = wt[id] + pre_trained_context[s[i]]
        wt[id] = [x / len(s) for x in wt[id]]

    return wt


def getStatsFromFile(path):
    polarity_vector = []
    with open(path, "r") as fd:
        lines = fd.read().splitlines()
        size = len(lines) / 3
        for i in range(0, len(lines), 3):
            # polarity
            polarity_vector.append(lines[i + 2].strip().split()[0])
    return size, polarity_vector


def loadHyperData(config, loadData, percentage=0.8):
    FLAGS = config

    if loadData:
        """Splits a file in 2 given the `percentage` to go in the large file."""
        random.seed(12345)
        with open(FLAGS.train_path, 'r') as fin, \
                open(FLAGS.hyper_train_path, 'w') as foutBig, \
                open(FLAGS.hyper_eval_path, 'w') as foutSmall:
            lines = fin.readlines()

            chunked = [lines[i:i + 3] for i in range(0, len(lines), 3)]
            random.shuffle(chunked)
            numlines = int(len(chunked) * percentage)
            for chunk in chunked[:numlines]:
                for line in chunk:
                    foutBig.write(line)
            for chunk in chunked[numlines:]:
                for line in chunk:
                    foutSmall.write(line)
        with open(FLAGS.train_svm_path, 'r') as fin, \
                open(FLAGS.hyper_svm_train_path, 'w') as foutBig, \
                open(FLAGS.hyper_svm_eval_path, 'w') as foutSmall:
            lines = fin.readlines()

            chunked = [lines[i:i + 4] for i in range(0, len(lines), 4)]
            random.shuffle(chunked)
            numlines = int(len(chunked) * percentage)
            for chunk in chunked[:numlines]:
                for line in chunk:
                    foutBig.write(line)
            for chunk in chunked[numlines:]:
                for line in chunk:
                    foutSmall.write(line)

    # get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile(FLAGS.hyper_train_path)
    test_size, test_polarity_vector = getStatsFromFile(FLAGS.hyper_eval_path)

    return train_size, test_size, train_polarity_vector, test_polarity_vector


def loadCrossValidation(config, split_size, augment_data, load=True):
    FLAGS = config
    if load:
        words, svmwords, sent, words_t, sent_t = [], [], [], [], []

        with open(FLAGS.train_path, encoding='cp1252') as f, open(FLAGS.remaining_test_path, encoding='cp1252') as test:
            lines = f.readlines()
            t_lines = test.readlines()
            if augment_data:
                print(len(lines))
                lines *= FLAGS.original_multiplier
                aug_lines = io.open(FLAGS.augmentation_file_path, 'r', encoding='utf-8').readlines()
                lines.extend(aug_lines)
                print(len(lines))
                print(len(aug_lines))
            for i in range(0, len(lines), 3):
                words.append([lines[i], lines[i + 1], lines[i + 2]])
                sent.append(lines[i + 2].strip().split()[0])
            for i in range(0, len(t_lines), 3):
                words_t.append([t_lines[i], t_lines[i+1], t_lines[i+2]])
                sent_t.append(lines[i + 2].strip().split()[0])
            words = np.asarray(words)
            sent = np.asarray(sent)
            words_t = np.asarray(words_t)
            sent_t = np.asarray(sent_t)

            i = 0
            kf = StratifiedKFold(n_splits=split_size, shuffle=True, random_state=12345)
            for train_idx, val_idx in kf.split(words, sent):
                words_1 = words[train_idx]
                words_2 = words[val_idx]
                # svmwords_1 = svmwords[train_idx]
                # svmwords_2 = svmwords[val_idx]
                with open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_' + str(
                        i) + '.txt', 'w') as train, \
                        open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_' + str(
                            i) + '.txt', 'w') as val:
                    for row in words_1:
                        train.write(row[0])
                        train.write(row[1])
                        train.write(row[2])
                i += 1
            i=0
            for train_idx, val_idx in kf.split(words_t, sent_t):
                words_1 = words[train_idx]
                words_2 = words[val_idx]
                # svmwords_1 = svmwords[train_idx]
                # svmwords_2 = svmwords[val_idx]
                with open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_' + str(
                        i) + '.txt', 'w') as train, \
                        open("data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_' + str(
                            i) + '.txt', 'w') as val:
                    for row in words_1:
                        val.write(row[0])
                        val.write(row[1])
                        val.write(row[2])
                i += 1
        # get statistic properties from txt file
    train_size, train_polarity_vector = getStatsFromFile(
        "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_train_0.txt')
    test_size, test_polarity_vector = [], []
    for i in range(split_size):
        test_size_i, test_polarity_vector_i = getStatsFromFile(
            "data/programGeneratedData/crossValidation" + str(FLAGS.year) + '/cross_val_' + str(i) + '.txt')
        test_size.append(test_size_i)
        test_polarity_vector.append(test_polarity_vector_i)

    return train_size, test_size, train_polarity_vector, test_polarity_vector
