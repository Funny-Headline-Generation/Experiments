import os
import re
import numpy as np
import nltk
import stanza
import spacy_udpipe
import spacy
import json

from gensim import matutils
from ast import literal_eval

from scipy.sparse import csr_matrix, save_npz, load_npz
from collections import Counter
from copy import copy
from tqdm import tqdm_notebook

from gensim.models import KeyedVectors
from ast import literal_eval

def two_way_map(l):
    id2elem = [i for i in l]
    elem2id = {id2elem[i]:i for i in range(len(id2elem))}
    return id2elem, elem2id

def diag_len(l):
    f = lambda x: int((x*(x+1))/2)
    return f(len(l))


class Tokenizer:
    def __init__(self, method='nltk'):
        self.method = method
        self.pipeline = None

        if method == 'stanford_upos':
            self.pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos')
        if method == 'stanford_nltk':
            self.pipeline = stanza.Pipeline(lang='en', processors='pos')
        elif method == 'udpipe_upos':
            self.pipeline = spacy_udpipe.load("en")
        elif method == 'spacy_upos':
            self.pipeline = spacy.load("en_core_web_sm")

    def __call__(self, text):
        if self.method == 'nltk':
            return self.nltk_tokenize(text)
        elif self.method == 'stanford_upos':
            return self.stanford_upos_tokenize(text)
        elif self.method == 'stanford_nltk':
            return self.stanford_upos_tag(self.nltk_tokenize(text))
        elif self.method == 'udpipe_upos':
            return self.udpipe_upos_tokenize(text)
        elif self.method == 'spacy_upos':
            return self.spacy_upos_tokenize(text)

    def nltk_tokenize(self, text):
        text = nltk.sent_tokenize(text)
        text = [nltk.word_tokenize(sent) for sent in text]
        return text
    
    def stanford_upos_tokenize(self,text):
        text = self.pipeline(text)
        return [[f"{token.text}_{token.upos}" for token in sent.words] for sent in text.sentences]
    
    def stanford_upos_tag(self, text):
        return NotImplemented
    
    def udpipe_upos_tokenize(self, text):
        text = self.pipeline(text)
        return [[f"{token.lemma}_{token.upos}" for token in sent.words] for sent in text.sentences]
    
    def spacy_upos_tokenize(self, text):
        sents = [self.pipeline(sent) for sent in nltk.sent_tokenize(text)]
        return [[f"{token.lemma_}_{token.pos_}" for token in sent] for sent in sents]

class NGramMatrix:
    def __init__(self, n, tokenizer):
        self.n = n
        self.tokenizer = tokenizer
        self.N = 0
        self.counts = None
        self.ngram_matrix = None
        self.id2key, self.key2id = None, None
    
    def save(self, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)
        for key,val in self.__dict__.items():
            if type(val) == dict or type(val) == list:
                with open(os.path.join(directory, f"{key}.json"),'w',encoding='utf-8') as outp:
                    if type(val) == dict:
                        json.dump({str(a):b for a,b in val.items()}, outp, ensure_ascii=False)
                    elif type(val) == list:
                        json.dump(val, outp, ensure_ascii=False)
            elif type(val) == csr_matrix:
                save_npz(os.path.join(directory, f"{key}.npz"), val)

    def load(self, directory):
        for file in os.listdir(directory):
            fname, ext = file.split('.')
            if ext == 'json':
                with open(os.path.join(directory, file), 'r', encoding='utf-8') as inp:
                    self.__setattr__(fname, json.load(inp))
                    if type(getattr(self,fname)) == dict:
                        d = getattr(self,fname)
                        for key in d:
                            if re.match('\(\"[a-zA-Z0-9_]+\", *?\"[a-zA-Z0-9_]+\"\)',key):
                                key_parsed = literal_eval(key)
                                d[key_parsed] = d[key]
                                d.pop(key)
                    elif type(getattr(self,fname)) == dict:
                        l = getattr(self,fname)
                        for i in range(len(l)):
                            if type(l[i]) == list:
                                l[i] = tuple(l[i])
            elif ext == 'npz':
                self.__setattr__(fname, load_npz(os.path.join(directory, file)))

    def train(self, corpus):
        print('Tokenizing...')

        if type(corpus) == str:
            corpus = self.tokenizer(corpus)
        else:
            corpus = [sent for text in tqdm_notebook(corpus,
            total=len(corpus))for sent in self.tokenizer(text)]

        self.N = sum(len(sent) for sent in corpus)
        self.counts = Counter(word for sent in corpus for word in sent)
        n_cols = len(self.counts)
        n_rows = n_cols

        for i in range(2, self.n+1):
            self.counts += Counter(tuple(sent[j:j+i]) for sent in corpus \
                for j in range(len(sent)-i))
            if i == self.n-1:
                n_rows = len(self.counts)

        self.id2key, self.key2id = two_way_map(self.counts)
        values, row_ids, col_ids = np.zeros(len(self.counts)-n_cols),\
                                   np.zeros(len(self.counts) - n_cols),\
                                   np.zeros(len(self.counts)-n_cols)
        i = 0

        print('Calculating n-gram frequencies...')

        count_iter = tqdm_notebook(self.counts, total=len(self.counts))

        for key in count_iter:
            if type(key) == tuple:
                values[i] = self.counts[key]
                if len(key[:-1]) == 1:
                    row_ids[i] = self.key2id[key[0]]
                else:
                    row_ids[i] = self.key2id[key[:-1]]
                col_ids[i] = self.key2id[key[-1]]
                i += 1

        self.ngram_matrix = csr_matrix((values, (row_ids, col_ids)),
                                        shape=(n_rows, n_cols))

class Metric:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, ngram_matrix, item_counts, key2id, N, verbose):
        values, row_ids, col_ids = np.zeros(len(ngram_matrix.data)), \
                                   np.zeros(len(ngram_matrix.data)), \
                                   np.zeros(len(ngram_matrix.data))
        i = 0

        if verbose:
            item_iter = item_counts
        else:
            item_iter = tqdm_notebook(item_counts, total=len(item_counts))
        
        for ab in item_iter:
            if type(ab) == tuple:
                if len(ab[:-1]) == 1:
                    elem_a = ab[0]
                else:
                    elem_a = ab[:-1]
                elem_b = ab[-1]
                values[i] = self.metric(ab=item_counts[ab],
                                        a=item_counts[elem_a],
                                        b=item_counts[elem_b],
                                        N=N)
                row_ids[i] = key2id[elem_a]
                col_ids[i] = key2id[elem_b]
                i += 1
        return csr_matrix((values, (row_ids, col_ids)),
                          shape=ngram_matrix.shape)

class PMI(Metric):
    def __init__(self):
        pmi = lambda ab,a,b,N: np.log(ab) + np.log(N) - np.log(a) - np.log(b)
        super().__init__(pmi)


class CollocateMatrix(NGramMatrix):
    def __init__(self, n, tokenizer, metric):
        super().__init__(n, tokenizer)
        self.metric = metric
        self.collocate_matrix = None

    def train(self, corpus):
        print('Training N-gram matrix...')
        super().train(corpus)
        print('Calculating metric...')
        self.collocate_matrix = self.metric(self.ngram_matrix,
                                            self.counts,
                                            self.key2id,
                                            self.N,
                                            verbose=True)

    def get_collocations(self, text, thresh=1.0):
        text = [word for sent in self.tokenizer(text) for word in sent]
        outp = []

        for ngram_range in range(2, self.n+1):
            for token_id in range(len(text)-ngram_range+1):
                if ngram_range == 2:
                    key = text[token_id]
                else:
                    key = tuple(text[token_id:token_id+ngram_range-1])
                val = text[token_id+ngram_range-1]
                strength = self.collocation_strength(key, val)
                if ngram_range == 2:
                    key = (key,)
                if strength is not None:
                    if strength >= thresh:
                        outp.append((
                            (token_id, token_id+ngram_range-1),
                            key + (val,),
                            strength
                        ))
        
        return outp
    
    def collocation_strength(self, key, val):
        if key in self.key2id and val in self.key2id:
            key_id, val_id = self.key2id[key], self.key2id[val]
            strength = self.collocate_matrix[key_id].toarray().squeeze()[val_id]
            return strength
        else:
            return None


def most_distant_same_pos(word, model,
thresh=0, max_count=0, pos_after_word=True):
    '''
    Find most distant words to given
    assuming the model is binary (and therefore normalized)
    and includes pos-tagging (in form of word_POS)
    '''
    to_dist = lambda x: 1 - (x + 1)/2
    word_vec = matutils.unitvec(model[word]).astype(np.float32)
    dists = np.dot(model.vectors, word_vec)
    ## map cosines to (0,1) range
    ## where higher values indicate higher distance:
    dists = to_dist(dists)

    sorted_dist_ids = matutils.argsort(dists, reverse=True)
    
    if pos_after_word:
        word_distances = [
            (model.index2entity[word_id], float(dists[word_id]))
            for word_id in sorted_dist_ids \
                if model.index2entity[word_id].endswith(word.split('_')[-1])\
                    and float(dists[word_id]) > thresh
        ]
    else:
        raise NotImplementedError

    if max_count:
        word_distances = word_distances[:max_count]
    
    return word_distances