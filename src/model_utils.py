import numpy as np

from numpy import linalg as LA
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances


def classify_label(vectors, classifier):
    ''' Classify inputs and return labels '''
    return classifier.predict(vectors)


def classify_prob(vectors, classifier, target_class=None):
    ''' Classify inputs and return probability '''
    if target_class is None:
        return classifier.predict_proba(vectors)
    else:
        idx = list(classifier.classes_).index(target_class)
        return classifier.predict_proba(vectors)[:, idx]


def normalize(x):
    return x / np.sum(np.abs(x))


def ontology_based_weights(x_vectors, z_vectors, ontology, vocab_list,
                           kernel_width=25,
                           metric='cosine', cosine_scale=100.0):
    '''
    Compute the weights (kernel function) based on distance between z and x,
    Differs in ontology will increase distance
    Using their vector represenation
    '''
    if metric == 'cosine':
        distances = paired_cosine_distances(x_vectors, z_vectors) * 100.0
    elif metric == 'euclidean':
        distances = paired_euclidean_distances(x_vectors, z_vectors)
    else:
        raise NotImplementedError('Metric not implemented')
    return np.sqrt(np.exp(- (np.square(distances)) / np.square(kernel_width)))


def norm_min_max(a):
    norm = [(float(i) - min(a)) / (max(a) - min(a)) for i in a]
    return norm


def score_funct(new_vect, ori_vect):
    score = LA.norm(new_vect - ori_vect)
    return score


def stack_info_pl(_sentence, _tokenizer, _language_model):
    word_vectors = _language_model.wv
    if _sentence == 'nan':
        result = [0] * 300
    else:
        result = []
        word_tokens = _tokenizer.tokenize(_sentence)  # remove punctuation
        for word in word_tokens:
            if word in word_vectors.vocab:
                result.append(_language_model[word])
            else:
                tmp = np.asarray([0] * 300)
                result.append(tmp)
        if len(result) != 0:
            result = np.vstack(result)
            result = np.sum(result, axis=0)
        else:
            result = np.asarray([0] * 300)
    return result
