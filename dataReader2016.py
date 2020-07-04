import os
import json
import xml.etree.ElementTree as ET
from collections import Counter
import string
import en_core_web_sm

n_nlp = en_core_web_sm.load()
# import spacy
import nltk
import re
import numpy as np
import data_augmentation
from config import *
import random
from tqdm import tqdm
import io


def window(iterable, size):  # stack overflow solution for sliding window
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


def _get_data_tuple(sptoks, asp_termIn, label):
    # Find the ids of aspect term
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks, len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break

    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = -1
    elif label == 'neutral':
        lab = 0
    elif label == "positive":
        lab = 1
    else:
        raise ValueError("Unknown label: %s" % lab)

    return pos_info, lab


"""
This function reads data from the xml file

Iput arguments:
@fname: file location
@source_count: list that contains list [<pad>, 0] at the first position [empty input]
and all the unique words with number of occurences as tuples [empty input]
@source_word2idx: dictionary with unique words and unique index [empty input]
.. same for target

Return:
@source_data: list with lists which contain the sentences corresponding to the aspects saved by word indices 
@target_data: list which contains the indices of the target phrases: THIS DOES NOT CORRESPOND TO THE INDICES OF source_data 
@source_loc_data: list with lists which contains the distance from the aspect for every word in the sentence corresponding to the aspect
@target_label: contains the polarity of the aspect (0=negative, 1=neutral, 2=positive)
@max_sen_len: maximum sentence length
@max_target_len: maximum target length

"""


def read_data_2016(fname, source_count, source_word2idx, target_count, target_phrase2idx, file_name, augment_data,
                   augmentation_file):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    # parse xml file to tree
    tree = ET.parse(fname)
    root = tree.getroot()

    outF = open(file_name, "w")
    augmF = io.open(augmentation_file, "w", encoding='utf-8') if augmentation_file else None

    # save all words in source_words (includes duplicates)
    # save all aspects in target_words (includes duplicates)
    # finds max sentence length and max targets length
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    target_phrases = []

    augmenter = data_augmentation.Augmentation(eda_type=FLAGS.EDA_type)
    augmented_sentences = []

    countConfl = 0
    category_counter = []
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        for sp in sptoks:
            source_words.extend([''.join(sp).lower()])
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)
        for opinions in sentence.iter('Opinions'):
            for opinion in opinions.findall('Opinion'):
                if opinion.get("polarity") == "conflict":
                    countConfl += 1
                    continue
                asp = opinion.get('target')
                if asp != 'NULL':
                    aspNew = re.sub(' +', ' ', asp)
                    t_sptoks = nltk.word_tokenize(aspNew)
                    category_counter.append(opinion.get('category'))
                    for sp in t_sptoks:
                        target_words.extend([''.join(sp).lower()])
                    target_phrases.append(' '.join(sp for sp in t_sptoks).lower())
                    if len(t_sptoks) > max_target_len:
                        max_target_len = len(t_sptoks)

    counted_cats = Counter(category_counter)
    print('category distribution for {} : {}'.format(file_name, counted_cats))
    if augment_data:
        category_sorter = {}  # For random swap of targets between sentences
        for i in counted_cats.keys():
            category_sorter[i] = []  # initialize as empty list
        print('starting data augmentation')
        for sentence in tqdm(root.iter('sentence')):
            sent = sentence.find('text').text
            sentenceNew = re.sub(' +', ' ', sent)
            for opinions in sentence.iter('Opinions'):
                for opinion in opinions.findall('Opinion'):
                    asp = opinion.get('target')
                    category = opinion.get('category')
                    polarity = opinion.get('polarity')
                    if asp != 'NULL':
                        aspNew = re.sub(' +', ' ', asp)
                        category_sorter[category].append(
                            {'sentence': sentenceNew, 'aspect': aspNew, 'polarity': polarity})
                        aug_sent, aug_asp = augmenter.augment(sentenceNew, aspNew)
                        aug_tok = nltk.word_tokenize(aug_asp)
                        for sp in aug_tok:
                            target_words.extend([''.join(sp).lower()])
                        for a_s in aug_sent:
                            sptoks = nltk.word_tokenize(a_s)
                            for sp in sptoks:
                                source_words.extend([''.join(sp).lower()])
                            augmented_sentences.append({'sentence': a_s,
                                                        'aspect': aspNew,
                                                        'category': category,
                                                        'polarity': polarity})
        for category in category_sorter.keys():
            if FLAGS.EDA_swap == 0 or FLAGS.EDA_type == 'original':  # we don't swap
                break
            sentences_same_cat = category_sorter[category]  # All sentences with the same category
            indices = np.array(range(len(sentences_same_cat)-1))
            random.shuffle(indices)  # Random index with which we shuffle
            for _ in range(FLAGS.EDA_swap):
                for i, j in tqdm(zip(*[iter(indices)] * 2)):
                    adder = 0
                    while sentences_same_cat[i].get('aspect') == sentences_same_cat[(j + adder) % len(indices)].get('aspect') and adder<100:  # happens more than you think
                        adder += 1
                    sent1, sent2 = augmenter.swap_targets(sentences_same_cat[i], sentences_same_cat[(j + adder) % len(indices)])
                    for sent in [sent1, sent2]:
                        sptoks = nltk.word_tokenize(sent['sentence'])
                        for sp in sptoks:
                            source_words.extend([''.join(sp).lower()])
                    augmented_sentences.extend([sent1, sent2])

            random.shuffle(sentences_same_cat)
            y = [sentences_same_cat[i * 2: (i + 1) * 2] for i in range(5)]

    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    for phrase, _ in target_count:
        if phrase not in target_phrase2idx:
            target_phrase2idx[phrase] = len(target_phrase2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()

    # collect output data (match with source_word2idx) and write to .txt file
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        if len(sptoks) != 0:
            idx = []
            for sptok in sptoks:
                idx.append(source_word2idx[''.join(sptok).lower()])
            for opinions in sentence.iter('Opinions'):
                for opinion in opinions.findall('Opinion'):
                    if opinion.get("polarity") == "conflict": continue
                    asp = opinion.get('target')
                    if asp != 'NULL':  # removes implicit targets
                        aspNew = re.sub(' +', ' ', asp)
                        t_sptoks = nltk.word_tokenize(aspNew)
                        source_data.append(idx)
                        outputtext = ' '.join(sp for sp in sptoks).lower()
                        outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                        outputtext = outputtext.replace(outputtarget, '$T$')
                        outF.write(outputtext)
                        outF.write("\n")
                        outF.write(outputtarget)
                        outF.write("\n")
                        pos_info, lab = _get_data_tuple(sptoks, t_sptoks, opinion.get('polarity'))
                        pos_info = [(1 - (i / len(idx))) for i in pos_info]
                        source_loc_data.append(pos_info)
                        targetdata = ' '.join(sp for sp in t_sptoks).lower()
                        target_data.append(target_phrase2idx[targetdata])
                        target_label.append(lab)
                        outF.write(str(lab))
                        outF.write("\n")

    outF.close()
    # Write augmented sentences
    if augment_data:
        for aug_sen in augmented_sentences:
            sptoks = nltk.word_tokenize(aug_sen['sentence'])
            if len(sptoks) != 0:
                idx = []
                for sptok in sptoks:
                    try:
                        idx.append(source_word2idx[''.join(sptok).lower()])
                    except KeyError:
                        raise KeyError('Word {} is not found in the word2index file'.format(sptok))
                asp = aug_sen['aspect']
                if asp != 'NULL':  # removes implicit targets
                    aspNew = re.sub(' +', ' ', asp)
                    t_sptoks = nltk.word_tokenize(aspNew)
                    source_data.append(idx)
                    outputtext = ' '.join(sp for sp in sptoks).lower()
                    outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                    outputtext = outputtext.replace(outputtarget, '$T$')
                    augmF.write(outputtext)
                    augmF.write("\n")
                    augmF.write(outputtarget)
                    augmF.write("\n")
                    pos_info, lab = _get_data_tuple(sptoks, t_sptoks, aug_sen.get('polarity'))
                    pos_info = [(1 - (i / len(idx))) for i in pos_info]
                    source_loc_data.append(pos_info)
                    targetdata = ' '.join(sp for sp in t_sptoks).lower()
                    target_data.append(target_phrase2idx[targetdata])
                    target_label.append(lab)
                    augmF.write(str(lab))
                    augmF.write("\n")
        augmF.close()
    print("Read %s aspects from %s" % (len(source_data), fname))
    print(countConfl)
    print("These are the augmentations that are done for ".format(fname), augmenter.counter)
    ct = augmenter.counter
    return [source_data, source_loc_data, target_data, target_label, max_sent_len, source_loc_data,
            max_target_len], ct
