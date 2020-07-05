# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou, adjusted by Tomas Liesting

import random
from random import shuffle

random.seed(1)

# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', '']

# cleaning up text
import re


def get_only_chars(line):
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm$- ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

# for the first time you use wordnet
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
import nltk
from pywsd import simple_lesk

def synonym_replacement(words, aspect, n, adjusted):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words and word != '$t$']))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        if not adjusted:
            synonyms = get_synonyms(random_word)
        else:
            synonyms = get_synonyms_adjusted(words, aspect, random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            temp = []
            replaced = False
            for word in new_words:
                if word == random_word and not replaced:
                    temp.append(synonym)
                    replaced = True
                else:
                    temp.append(word)
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
            new_words = temp
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def get_synonyms_adjusted(words, aspect, random_word):
    pos_tags = nltk.pos_tag(words)
    for word, func in pos_tags:
        if word == random_word and not get_wordnet_pos(func):
            return []
        elif word == random_word:
            with_asp = [x if x != '$t$' else aspect for x in words]
            meaning = simple_lesk(' '.join(with_asp), random_word, pos=get_wordnet_pos(func))
    synonyms = []
    if meaning:
        for syn in meaning.lemma_names():
            synonym = syn.lower()
            synonyms.append(synonym)
        if random_word in synonyms:
            synonyms.remove(random_word)
    return synonyms


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p or '$t$' in word:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n, adjusted=False):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words, adjusted=False):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word) if not adjusted else get_synonyms_adjusted(new_words, random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, aspect, alpha_sr=0, alpha_ri=0, alpha_rs=0, p_rd=0, percentage=.2, adjusted=False, counter=None):
    # sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    n_sr = max(1, int(percentage * num_words))
    n_ri = max(1, int(percentage * num_words))
    n_rs = max(1, int(percentage * num_words))

    # sr
    for _ in range(alpha_sr):
        a_words = synonym_replacement(words, aspect, n_sr, adjusted)
        augmented_sentences.append(' '.join(a_words))
        counter['synonym replacement']+=1

    # ri
    for _ in range(alpha_ri):
        a_words = random_insertion(words, n_ri, adjusted)
        augmented_sentences.append(' '.join(a_words))
        counter['random insertion'] += 1
    # rs
    if not adjusted:
        for _ in range(alpha_rs):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))
            counter['random swap'] += 1
    # rd
    for _ in range(p_rd):
        a_words = random_deletion(words, percentage)
        augmented_sentences.append(' '.join(a_words))
        counter['random deletion'] += 1

    augmented_sentences = [sentence for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    return augmented_sentences
