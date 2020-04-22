import re
import string
import numpy as np


def change_format(data, label):
    idx = np.arange(0, 1)
    x = [data[i] for i in idx]
    y = [label[i] for i in idx]
    return x, y


def check_concepts(tweet):
    word_tokens = tweet.split()
    in_onto_words = []
    for _i in range(len(word_tokens)):
        if word_tokens[_i] in ontology[0]:
            in_onto_words.append(word_tokens[_i])
    return in_onto_words


def concatenate_list_data(num, _list):
    result = 'Important score: ' + str(num) + ' ---- Explanation: '
    for element in _list:
        result += str(element) + ''
    return result


def raw_tweet_change_punc(raw_tweet):
    change_tweet = raw_tweet.replace('!', '.')  # Change ! to .
    change_tweet = change_tweet.replace('?', '.')  # Change ? to .
    return change_tweet


def rem_punc(tweet):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_tweet = regex.sub(' ', tweet)
    return clean_tweet


def is_verb(tag):
    if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'verb']:
        return 1
    else:
        return 0


def is_adj(tag):
    if tag in ['JJ', 'JJR', 'JJS', 'adj']:
        return 1
    else:
        return 0


def is_prep(tag):
    if tag in ['IN', 'TO', 'CC']:
        return 1
    else:
        return 0


def w_type(tag):
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:  # noun
        return 'noun'
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:  # verb
        return 'verb'
    elif tag in ['JJ', 'JJR', 'JJS']:  # adjective
        return 'adj'
    else:
        return 0


def build_local_vocab_bow_modify(target_tweet):
    ''' Build vocab out of target_tweet
    Compare to un-modified version, here we accept repetition
    Because of local fidelity characteristic, the same word but appears in different position can have
    different score/treated differently
    since depend on the position they are and their neighbour words '''
    local_vocab_list = []
    local_vocab_dict = {}
    words = target_tweet.strip().split()
    for word in words:
        idx = len(local_vocab_list)  # word_id
        local_vocab_dict[word] = idx
        local_vocab_list.append(word)
    return local_vocab_list, local_vocab_dict
