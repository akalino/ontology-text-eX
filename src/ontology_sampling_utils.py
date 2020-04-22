import numpy as np
import itertools
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences

from copy import deepcopy

from text_processing_utils import rem_punc, is_prep, build_local_vocab_bow_modify


def normal_sampling(z_prime, i, sample_normal):
    rnd_normal = np.random.rand()
    if rnd_normal > sample_normal:
        z_prime[i] = 0
    return z_prime


def prime2word(z_str_original, z_prime):
    z_str = []
    for i in range(len(z_str_original)):
        if z_prime[i] == 1:
            z_str.append(z_str_original[i])
    return z_str


def occurrences(s, lst):
    return (i for i, e in enumerate(lst) if e == s)


def sampling_process(len_tweet, word_tokens, pos, start_id, position, local_fid, ontology,
                     in_onto_concepts, isolated_concept, abstract, sample_onto,
                     sample_normal, min_words_per_tweet):
    st = PorterStemmer()
    tuples = []
    while True:  # Run until a qualified sample found (which has more than 1 word)
        z_prime = [1] * len_tweet
        if len_tweet > local_fid:
            segment = [word_tokens[i] for i in range(start_id, start_id + local_fid + 1)]
        else:
            segment = deepcopy(word_tokens)
        n = len(segment)
        # --------***** Inside the segment *****-------- #
        # Find all tuples and sample them in the segment
        rnd = np.random.rand()
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                a1 = segment[i]
                a2 = segment[j]
                real_i = start_id + i
                real_j = start_id + j
                a1 = st.stem(a1)
                a2 = st.stem(a2)
                if (a1 in ontology[0]) and (a2 in ontology[0]):  # Case 1: Two words appear in the ontology
                    a1_c = in_onto_concepts[a1]
                    a2_c = in_onto_concepts[a2]
                    if (len(a1_c) == 1) and (
                            a1_c == a2_c):  # Case 1.1: 2 words in the same main concept are not in a tuple
                        # print('Not in a tuple')
                        z_prime = normal_sampling(z_prime, real_i, sample_normal)
                        z_prime = normal_sampling(z_prime, real_j, sample_normal)
                    else:
                        # Case 1.2: words belong to different concepts in the ontology. Have to consider different cases
                        # Avoid the case of isolated concepts, e.g., MedicalCondition
                        a1_c = [x for x in a1_c if x not in isolated_concept]
                        a2_c = [x for x in a2_c if x not in isolated_concept]
                        # Put sent_based_concept here !!!!!!!!!!
                        idx_pos_i = real_i + position
                        idx_pos_j = real_j + position
                        pos_i = pos[idx_pos_i][1]
                        pos_j = pos[idx_pos_j][1]
                        a1_c_filter = a1_c
                        a2_c_filter = a2_c
                        # a1_c_filter = filter_concept(a1_c,pos_i,concept_assign_dict)
                        # a2_c_filter = filter_concept(a2_c,pos_j,concept_assign_dict)
                        if (len(a1_c_filter) == 1) and (a1_c_filter == a2_c_filter):
                            # Case 1.1: 2 words in the same main concept are not in a tuple
                            # print('Not in a tuple')
                            z_prime = normal_sampling(z_prime, real_i, sample_normal)
                            z_prime = normal_sampling(z_prime, real_j, sample_normal)
                        else:
                            edge_a1_c = [abstract[k] for k in a1_c_filter]
                            edge_a2_c = [abstract[k] for k in a2_c_filter]
                            # To concatenate all edges (list) & remove redundance
                            edge_a1_c = set(sum(edge_a1_c, []))
                            edge_a2_c = set(sum(edge_a2_c, []))
                            if len(list(edge_a1_c & edge_a2_c)) > 0:
                                # Case 1.2.1: A tuple is FOUND! when there is at least 1 sharing
                                # edge (a direct connection) between concepts
                                # print('In a tuple')
                                tuples.append([real_i, real_j])
                                if rnd > sample_onto:  # inactive/remove words
                                    z_prime[real_i] = 0
                                    z_prime[real_j] = 0
                            else:
                                # Case 1.2.2: Words appear in the ontology, but belong to undirect
                                # connection (no sharing edges between them)
                                # print('Not in a tuple')
                                z_prime = normal_sampling(z_prime, real_i, sample_normal)
                                z_prime = normal_sampling(z_prime, real_j, sample_normal)
                else:  # Case 2: At least 1 word is not in the ontology
                    # print('Word not in ontology')
                    z_prime = normal_sampling(z_prime, real_i, sample_normal)
                    z_prime = normal_sampling(z_prime, real_j, sample_normal)
        # --------***** Outside the segment *****-------- #
        if len_tweet > local_fid:
            range_in = list(range(start_id, start_id + local_fid + 1))
            range_out = [x for x in range(len(word_tokens)) if x not in range_in]
            for i in range_out:
                z_prime = normal_sampling(z_prime, i, sample_normal)
        if sum(z_prime) >= min_words_per_tweet:
            z_str = prime2word(word_tokens, z_prime)
            tuples = [list(x) for x in set(tuple(x) for x in tuples)]
            tmp = []
            for p in tuples:
                tmp.append([m + position for m in p])
            tuples = deepcopy(tmp)
            break
    return z_str, z_prime, tuples


def ontology_based_sample_z(target_tweet, pos, ontology, abstract_concepts, classifier,
                            cause_list, concept_assign_dict, vocab_cc, max_length,
                            local_fid, no_repeat, sample_normal, sample_onto, min_words_per_tweet):
    """
    Going to be the main function for ontology based sampling.

    :param target_tweet:
    :param pos:
    :param ontology:
    :param abstract_concepts:
    :param classifier:
    :param cause_list:
    :param concept_assign_dict:
    :param vocab_cc:
    :param max_length:
    :param local_fid:
    :param no_repeat:
    :param sample_normal:
    :param sample_onto:
    :param min_words_per_tweet:
    :return: Everything under the sun.
    """
    ''' Based on ontology, sample z from x '''
    # Conjunction words replacement. Fake punctuation
    st = PorterStemmer()
    rem_conj_tweet = deepcopy(target_tweet)
    tmp = word_tokenize(rem_conj_tweet)
    list_occ = []
    for i in range(len(cause_list)):
        occ = list(occurrences(cause_list[i], tmp))
        if len(occ) > 0:
            list_occ.append(list(occurrences(cause_list[i], tmp)))
    if len(list_occ) > 0:
        for i in list_occ:
            tmp[i[0]] = '.'
    rem_conj_tweet = ' '.join(tmp)
    rem_conj_tweet_remPunc = rem_punc(rem_conj_tweet)
    # Stemming tweet which removes conjuction (because, so, as...)
    tmp = word_tokenize(rem_conj_tweet)
    rem_conj_tweet_stem = [st.stem(word) for word in tmp]  # Stemming
    rem_conj_tweet_stem = ' '.join(rem_conj_tweet_stem)
    rem_conj_tweet_stem_remPunc = rem_punc(rem_conj_tweet_stem)
    # Stemming
    tmp = word_tokenize(target_tweet)
    target_tweet_stem = [st.stem(word) for word in tmp]  # Stemming
    target_tweet_stem = ' '.join(target_tweet_stem)
    # Vectorize tweets (strings) for classification
    target_tweet_stem_remPunc = rem_punc(target_tweet_stem)
    sen_tk = word_tokenize(rem_conj_tweet_remPunc)
    x_tmp = [vocab_cc[k] for k in sen_tk]
    target_vector = pad_sequences([x_tmp], maxlen=max_length, padding='post',
                                  truncating='post')  # truncate the post, and keep pre
    # TODO factor model call out
    target_predict = classifier.predict(target_vector, verbose=0)
    # Classify processed tweets (vectors)
    target_label = np.argmax(target_predict)

    target_tweet_remPunc = rem_punc(target_tweet)
    # TODO factor out bow model
    local_vocab_list, local_vocab_dict = build_local_vocab_bow_modify(rem_conj_tweet_remPunc)

    # ------- Traversal checking concepts on ontology ------- #
    # Find all ontology-based words in the target tweet & define their main concepts
    word_tokens_stem = target_tweet_stem_remPunc.split()  ########### STEMMING WORD TOKENS
    in_onto_concepts_stem = {}
    word_tokens = target_tweet_remPunc.split()
    in_onto_concepts = {}

    onto_list = list(ontology[0])
    concept_list = list(ontology[1])
    for i in range(len(word_tokens_stem)):
        if word_tokens_stem[i] in ontology[0]:
            all_inx = list(occurrences(word_tokens_stem[i], onto_list))
            tmp = []
            for j in all_inx:
                tmp.append(concept_list[j])
            concepts = list(set(tmp))
            in_onto_concepts_stem[word_tokens_stem[i]] = concepts  # dictionary
            in_onto_concepts[word_tokens[i]] = concepts  # dictionary

    # Find all edges/relationship between a concept with others.
    # The intuition is that 2 concepts share the same edge will be a tuple
    abstract = {}
    u = list(set(abstract_concepts[0]))
    isolated_concept = []
    for i in range(len(u)):
        ui = u[i]
        all_inx = list(occurrences(ui, abstract_concepts[0]))
        tmp = [abstract_concepts[1][all_inx[k]] for k in range(len(all_inx))]
        if tmp == ['iso']:
            isolated_concept.append(i)
        else:
            abstract[i] = tmp

    # ----------- *********** SAMPLING PROCESS BASED ON ONTOLOGY ************* ----------- #
    # Number of sampled tweets based on length of the tweet.
    # So, if it is a short tweet, no_repeat bigger to create more samples.
    # If it is a long tweet, no_repeat smaller but still enough samples for learning explainer
    or_tk = word_tokenize(target_tweet)
    rem_tk = word_tokenize(rem_conj_tweet)
    rem_pos = [i for i in range(len(or_tk)) if or_tk[i] != rem_tk[i]]

    sentences = rem_conj_tweet.split(' . ')
    sentences = [rem_punc(i) for i in sentences if len(i) != 0]
    word_tokens_list = []
    position_list = []
    count = 0
    if len(rem_pos) > 0:
        if rem_pos[0] == 0:
            count = 1
    for s_idx in range(len(sentences)):
        tmp = sentences[s_idx].split()
        if len(tmp) > 0:
            position_list.append(count)
            word_tokens_list.append(tmp)
            count += len(tmp)
            if count in rem_pos:
                count += 1

    # Remove . in pos tagging
    pos_remPunc = []
    pos_each = []
    pos_tmp = deepcopy(pos)
    if pos_tmp[len(pos_tmp) - 1] != '.':
        pos_tmp.append(('.', '.'))
    len_ = []
    for i in range(len(pos_tmp)):
        if pos_tmp[i][0] != '.':
            pos_each.append(pos_tmp[i])
        else:
            if len(pos_each) != 0:
                pos_remPunc.append(pos_each)
                len_.append(len(pos_each))
                pos_each = []
    # Find anchors
    start_anchors_w = ['not', 'no', 'illegal', 'against', 'without']
    in_anchor_w = []
    for i in range(len(sentences)):
        # print(i)
        s_tk = sentences[i].split()
        if '.' in s_tk:
            s_tk.remove('.')
        add_anchor = []
        for a in start_anchors_w:
            tmp = list(occurrences(a, s_tk))
            sorted_tmp = sorted(tmp)
            sorted_tmp.append(len(s_tk) - 1)
            if len(sorted_tmp) > 1:
                for t_i in range(len(sorted_tmp)):
                    add_anchor = []
                    t = [a]
                    t1 = sorted_tmp[t_i]
                    if t1 != len(s_tk) - 1:  # not the last word in the sentence
                        t2 = sorted_tmp[t_i + 1]
                        if t2 == len(s_tk) - 1:
                            t4 = len(s_tk)
                        else:
                            t4 = deepcopy(t2)
                        for k in range(t1 + 1, t4):  # t1+1 => t2
                            t.append(s_tk[k])
                            t3 = deepcopy(t)
                            end_pos = pos_remPunc[i][k][1]
                            if len(t3) < local_fid and is_prep(end_pos) == 0:
                                add_anchor.append(t3)
                        # Calculate the score and chose one anchor
                        if len(add_anchor) > 1:
                            x_tmp = [vocab_cc[k] for k in sen_tk]
                            ori_vector = pad_sequences([x_tmp], maxlen=max_length, padding='post',
                                                       truncating='post')  # truncate the post, and keep pre
                            ori_prob = np.max(classifier.predict(ori_vector, verbose=0), axis=1)
                            differ_prob_tmp = []
                            all_idx = list(range(len(sen_tk)))
                            for remove_idx in add_anchor:
                                rest_words = [j for j in sen_tk if j not in remove_idx]
                                x_tmp = [vocab_cc[j] for j in rest_words]
                                rest_vector = pad_sequences([x_tmp], maxlen=max_length, padding='post',
                                                            truncating='post')  # truncate the post, and keep pre
                                prob = np.max(classifier.predict(rest_vector, verbose=0), axis=1)
                                differ_prob_tmp.append(abs(prob - ori_prob).item())
                            max_ind = np.argmax(differ_prob_tmp)
                            chosen = add_anchor[max_ind]
                            in_anchor_w.append(' '.join(chosen))
                        else:
                            if len(add_anchor) > 0:
                                chosen = deepcopy(add_anchor[0])
                                in_anchor_w.append(' '.join(chosen))

    # Find sentences containing anchors
    anchor_position = []
    for i in range(len(sentences)):
        s_tk = sentences[i].split()
        if len(s_tk) > 0:
            for a in range(len(in_anchor_w)):
                a_tk = in_anchor_w[a].split()
                # no_a = len(a_tk)
                if len(a_tk) > 1:
                    for j in range(len(s_tk) - len(a_tk) + 1):
                        buffer = [s_tk[k] for k in range(j, j + len(a_tk))]
                        buffer_pos = [k for k in range(j, j + len(a_tk))]
                        buffer_pos = [k + position_list[i] for k in buffer_pos]
                        # print(buffer)
                        if buffer == a_tk:
                            anchor_position.append([in_anchor_w[a], buffer_pos, i])
                            break

    len_tweet = len(word_tokens)

    z_strs_list = []
    z_primes_list = []

    len_ = min(len_tweet, 2)
    for iter in range(no_repeat * len_):
        # This creates 1 sampled sentence
        z_strs_chunk = []
        z_primes_chunk = []
        tuples_chunk = []
        for s_idx in range(len(word_tokens_list)):
            word_tokens = word_tokens_list[s_idx]
            len_chunk = len(word_tokens)
            if len_chunk < 2:  # Obviously no tuples found (1) # Short chunk
                z_prime = [1] * len_chunk
                z_prime = normal_sampling(z_prime, 0, sample_normal)
                z_str = prime2word(word_tokens, z_prime)
                tuples = []
                z_strs_chunk.append([z_str])
                z_primes_chunk.append([z_prime])
                tuples_chunk.append([tuples])
            elif len_chunk > local_fid:  # (local_fid+1) or more # Long enough chunk
                z_strs_one = []
                z_primes_one = []
                tuples_one = []
                for start_id in range(0, len_chunk - local_fid, 3):
                    z_str, z_prime, tuples = sampling_process(len_chunk, word_tokens, pos, start_id,
                                                              position_list[s_idx], local_fid, ontology,
                                                              in_onto_concepts_stem, isolated_concept, abstract,
                                                              sample_onto, sample_normal, min_words_per_tweet)
                    z_strs_one.append(z_str)
                    z_primes_one.append(z_prime)
                    tuples_one.append(tuples)

                z_strs_chunk.append(z_strs_one)
                z_primes_chunk.append(z_primes_one)
                tuples_chunk.append(tuples_one)
            else:  # 2 to local_fid (2-3) # Short chunk
                z_str, z_prime, tuples = sampling_process(len_chunk, word_tokens, pos, 0, position_list[s_idx],
                                                          local_fid, ontology, in_onto_concepts_stem, isolated_concept,
                                                          abstract, sample_onto, sample_normal, min_words_per_tweet)
                z_strs_chunk.append([z_str])
                z_primes_chunk.append([z_prime])
                tuples_chunk.append([tuples])

        # Combination list
        tmp = list(itertools.product(*z_strs_chunk))
        for str_list in tmp:
            tmp2 = []
            for i in str_list:
                tmp2.extend(i)
            z_strs_list.append(tmp2)
        tmp = list(itertools.product(*z_primes_chunk))
        for str_list in tmp:
            tmp2 = []
            for i in str_list:
                tmp2.extend(i)
            z_primes_list.append(tmp2)
        tuples_list = []
        for str_list in tuples_chunk:
            tmp = []
            for i in str_list:
                if (len(i) != 0) and (i not in tmp):
                    tmp.append(i)
            tuples_list.append(tmp)

    return z_strs_list, z_primes_list, in_onto_concepts_stem, in_onto_concepts, \
           abstract, isolated_concept, tuples_list, target_label, target_vector, \
           local_vocab_list, local_vocab_dict, target_tweet_remPunc, target_tweet_stem_remPunc, \
           rem_conj_tweet, rem_conj_tweet_remPunc, rem_pos, anchor_position, position_list
