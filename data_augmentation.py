from EDA.code.eda import *
import nltk
from nltk.corpus import wordnet as wn
from config import *
from googletrans import Translator
from google.cloud import translate_v2 as translate
import pandas as pd
from scipy.stats import beta
import numpy as np
from utils import *

class Augmentation():
    def __init__(self, eda_type, need_mixup=False):
        self.adjusted = True if eda_type == 'adjusted' else False
        self.counter = {'random insertion': 0,
                        'random swap': 0,
                        'synonym replacement': 0,
                        'random deletion': 0,
                        'backtranslation': 0,
                        }
        self.translate_client = translate.Client()
        self.translator = Translator()
        if need_mixup:
            self.word_dict, self.w2v = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)
            self.start = len(self.w2v)
            self.counter = {'word_mixup': 0,
                            'sentence_mixup': 0}

    def augment(self, sent, asp):
        ret = []
        eda_sent, eda_asp = self.eda(sent, asp, adjusted=self.adjusted)
        backtrans_sent = self.backtranslation(sent, asp) if FLAGS.backtranslation_langs != 'None' else []
        ret.extend(backtrans_sent)
        ret.extend(eda_sent)
        return ret, asp

    def eda(self, sentence, aspect, adjusted=False):
        sent_adjusted = sentence.replace(aspect, '$t$')
        assert sent_adjusted != sentence, 'Something went wrong, the aspect "{}" cannot be found in "{}"'.format(aspect, sentence)
        augmented_sent = eda(sent_adjusted,
                             aspect,
                             alpha_ri=FLAGS.EDA_insertion,
                             alpha_rs=FLAGS.EDA_swap,
                             alpha_sr=FLAGS.EDA_replacement,
                             p_rd=FLAGS.EDA_deletion,
                             percentage=FLAGS.EDA_pct,
                             adjusted=adjusted,
                             counter=self.counter)
        augmented_with_aspect = []
        for sent in augmented_sent:
            augmented_with_aspect.append(sent.replace('$t$', aspect))
            assert sent != sent.replace('$t$', aspect), 'Something went wrong, the aspect "{}" cannot be found in "{}"'.format(
                "$t$", sent)
        return augmented_with_aspect, aspect

    def swap_targets(self, sentence1, sentence2):
        if sentence1['aspect'] == sentence2['aspect']:
            self.counter['random swap'] += 2
            return sentence1, sentence2
        renewed_sent1 = sentence1['sentence'].replace(sentence1['aspect'], sentence2['aspect'])

        assert renewed_sent1 != sentence1['sentence'], 'Something went wrong, the aspect "{}" cannot be found in "{}"'.format(sentence1['aspect'], sentence1['sentence'])

        renewed_sent2 = sentence2['sentence'].replace(sentence2['aspect'], sentence1['aspect'])
        assert renewed_sent2 != sentence2['sentence'], 'Something went wrong, the aspect "{}" cannot be found in "{}"'.format(sentence2['aspect'], sentence2['sentence'])

        aug_sent1 = {'sentence': renewed_sent1,
                     'aspect': sentence2['aspect'],
                     'polarity': sentence1['polarity']}

        aug_sent2 = {'sentence': renewed_sent2,
                     'aspect': sentence1['aspect'],
                     'polarity': sentence2['polarity']}
        self.counter['random swap'] += 2
        return aug_sent1, aug_sent2

    def backtranslation(self, sentence, aspect):
        adj_sent = sentence.replace(aspect, '$t$')
        assert adj_sent != sentence, 'Something went wrong, could not find {} in {}'.format(aspect, sentence)
        try:
            split = adj_sent.split('$t$')
            left_context, right_context = split[0], ' '.join(split[1:])
            right_context.replace('$t$', aspect)
        except ValueError:
            raise ValueError('something went wrong with splitting sentence {} on {}'.format(adj_sent, '$t$'))
        languages = FLAGS.backtranslation_langs.split(' ')
        backtranslated_sentences = []
        for lang in languages:
            translated_str = ''
            ctx = []
            for ctxt in [left_context, '$t$', right_context]:
                if ctxt != '$t$':
                    trans = self.translate_client.translate(ctxt, target_language=lang)
                    ctxt = self.translate_client.translate(trans['translatedText'], target_language='en')['translatedText']
                    ctx.append(trans['translatedText'])
                translated_str += ' ' + ctxt

            replaced = translated_str.replace('$t$', aspect)
            assert replaced != translated_str, 'Something went wrong, sentence "{}" does not have a target "$t$" anymore'.format(translated_str)
            try:
                df = pd.read_json(FLAGS.backtranslation_file)
            except ValueError:
                df = pd.DataFrame([])
            row = {'left_context': left_context,
                   'right_context': right_context,
                   'original_sentence': sentence,
                   'translated_sentence': ctx[0] + '$t$' + ctx[1],
                   'backtranslated_sentence': translated_str,
                   'language': lang}
            df.append(row, ignore_index=True)
            df.to_json(FLAGS.backtranslation_file)

            print('translated {} to {}'.format(adj_sent, replaced))
            backtranslated_sentences.append(replaced)
        return backtranslated_sentences

    def word_mixup(self, lcr_tuple1, lcr_tuple2):
        left1, len_left1, right1, len_right1, y1, target1, len_target1 = lcr_tuple1
        left2, len_left2, right2, len_right2, y2, target2, len_target2 = lcr_tuple2

        len_left = max(len_left1, len_left2)
        len_right = max(len_right1, len_right2)
        len_target = max(len_target1, len_target2)

        labda = np.random.beta(FLAGS.mixup_beta, FLAGS.mixup_beta)

        emb_left1 = np.array([self.w2v[x] for x in left1])
        emb_left2 = np.array([self.w2v[x] for x in left2])
        emb_right1 = np.array([self.w2v[x] for x in right1])
        emb_right2 = np.array([self.w2v[x] for x in right2])
        emb_target1 = np.array([self.w2v[x] for x in target1])
        emb_target2 = np.array([self.w2v[x] for x in target2])

        new_left = labda * emb_left1 + (1 - labda) * emb_left2
        new_right = labda * emb_right1 + (1 - labda) * emb_right2
        new_target = labda * emb_target1 + (1 - labda) * emb_target2
        new_y = labda * np.array(y1) + (1 - labda) * np.array(y2)
        left_indices = [0] * len(new_left)
        right_indices = [0] * len(new_right)
        target_indices = [0] * len(new_target)

        outF = open(FLAGS.embedding_path, "a")

        self.w2v = np.append(self.w2v, np.random.normal(0, 0.05, [len_left + len_target + len_right, 300]),
                        axis=0)  # lenleft+target+right new embeddings

        for i in range(len_left):
            outF.write('mixup_' + str(self.start))
            outF.write(" ")
            outF.write(' '.join(str(w) for w in new_left[i]))
            outF.write("\n")
            self.w2v[self.start] = new_left[i]
            left_indices[i] = self.start
            self.start += 1
        for i in range(len_target):
            outF.write('mixup_' + str(self.start))
            outF.write(" ")
            outF.write(' '.join(str(w) for w in new_target[i]))
            outF.write("\n")
            self.w2v[self.start] = new_target[i]
            target_indices[i] = self.start
            self.start += 1
        for i in range(len_right):
            outF.write('mixup_' + str(self.start))
            outF.write(" ")
            outF.write(' '.join(str(w) for w in new_right[i]))
            outF.write("\n")
            self.w2v[self.start] = new_right[i]
            right_indices[i] = self.start
            self.start += 1
        self.counter['word_mixup'] += 1
        return left_indices, len_left, right_indices, len_right, new_y, target_indices, len_target

    def sentence_mixup(self, first, second):
        labda = np.random.beta(FLAGS.mixup_beta, FLAGS.mixup_beta)
        left1, target1, right1 = first
        left2, target2, right2 = second

        new_left = labda*left1 + (1-labda) * left2
        new_target = labda * target1 + (1-labda)







