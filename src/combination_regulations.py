from copy import deepcopy

from text_processing_utils import w_type


def sublist(list_, sub_list):
    if all(x in list_ for x in sub_list):
        return True
    else:
        return False


def combine_tuple(tuples_list, pos, t1, t2, in_onto_concepts,
                  w_tk, concept_assign_dict, index):
    pos_t1 = pos[t1[index]][1]
    pos_t2 = pos[t2[index]][1]
    a1_c = in_onto_concepts[w_tk[t1[index]]]
    a2_c = in_onto_concepts[w_tk[t2[index]]]
    a1_c_filter = a1_c
    a2_c_filter = a2_c
    # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
    # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
    if a1_c_filter == a2_c_filter:  # A & B same concept
        t2.append(t1[index])
        combine = sorted(t2)
        if combine not in tuples_list:
            tuples_list.append(combine)
    else:
        if t1 not in tuples_list:
            tuples_list.append(t1)
        if t2 not in tuples_list:
            tuples_list.append(t2)
    return tuples_list


def filter_concept(a_c, pos_i, concept_assign_dict):
    pos = w_type(pos_i)
    sent = []
    for _i in a_c:
        concept = concept_assign_dict[_i]
        if pos in concept:
            sent.append(1)
        else:
            sent.append(0)
    true_concept = [a_c[_i] for _i in range(len(a_c)) if sent[_i] == 1]
    return true_concept


def rem_sublist(tuples_list):
    # Remove sub-tuples
    tuples_list_merge = deepcopy(tuples_list)
    for i in range(len(tuples_list)):
        for j in range(len(tuples_list)):
            if i != j:
                if sublist(tuples_list[i], tuples_list[j]):  # i is list, j is sublist
                    if tuples_list[j] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[j])
                if sublist(tuples_list[j], tuples_list[i]):  # j is list, i is sublist
                    if tuples_list[i] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[i])
    tuples_list = deepcopy(tuples_list_merge)
    return tuples_list


def rem_sublist_2(tuples_list, t_pos_all):
    # Remove sub-tuples
    t_pos_all_ = deepcopy(t_pos_all)
    tuples_list_merge = deepcopy(tuples_list)
    for i in range(len(tuples_list)):
        for j in range(len(tuples_list)):
            if i != j:
                if sublist(tuples_list[i], tuples_list[j]):  # i is list, j is sublist
                    if tuples_list[j] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[j])
                        t_pos_all_.remove(t_pos_all[j])
                if sublist(tuples_list[j], tuples_list[i]):  # j is list, i is sublist
                    if tuples_list[i] in tuples_list_merge:
                        tuples_list_merge.remove(tuples_list[i])
                        t_pos_all_.remove(t_pos_all[i])
    tuples_list = deepcopy(tuples_list_merge)
    t_pos_all = deepcopy(t_pos_all_)
    return tuples_list, t_pos_all


def combine_triple_2(tuples_list, t1, t2, intersect, in_onto_concepts, w_tk, concept_assign_dict, pos):
    p_t1_s = [i3 for i3 in range(len(t1)) if t1[i3] in intersect]
    p_t2_s = [i3 for i3 in range(len(t1)) if t2[i3] in intersect]
    if (p_t1_s == p_t2_s) and (p_t1_s == [0, 1]):
        # Case 1: same the beginning, e.g. ABC & ABD
        pos_t1 = pos[t1[len(p_t1_s)]][1]
        pos_t2 = pos[t2[len(p_t2_s)]][1]
        a1_c = in_onto_concepts[w_tk[t1[len(p_t1_s)]]]
        a2_c = in_onto_concepts[w_tk[t2[len(p_t2_s)]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        if a1_c_filter == a2_c_filter:  # C & D same concept
            if t1[len(p_t1_s)] < t2[len(p_t1_s)]:  # index of C < index of D in the sentence
                t1.append(t2[2])
                if t1 not in tuples_list:
                    tuples_list.append(t1)
            else:
                t2.append(t1[2])
                if t2 not in tuples_list:
                    tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == p_t2_s) and (p_t1_s == [1, 2]):
        # Case 2: same the end, e.g. ABC & DBC
        pos_t1 = pos[t1[0]][1]  # p_t1_s[0] - 1
        pos_t2 = pos[t2[0]][1]
        a1_c = in_onto_concepts[w_tk[t1[0]]]
        a2_c = in_onto_concepts[w_tk[t2[0]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        if a1_c_filter == a2_c_filter:  # A & D same concept
            if t1[0] < t2[0]:  # index of A < index of D in the sentence
                t1 = [t1[0]] + t2
                if t1 not in tuples_list:
                    tuples_list.append(t1)
            else:
                t2 = [t2[0]] + t1
                if t2 not in tuples_list:
                    tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == p_t2_s) and (p_t1_s == [0, 2]):
        # Case 2: same the beginning and the end, e.g. ABC & ADC
        pos_t1 = pos[t1[1]][1]
        pos_t2 = pos[t2[1]][1]
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        if a1_c_filter == a2_c_filter:  # B & D same concept
            if t1[1] < t2[1]:  # index of B < index of D in the sentence (ABDC)
                t1 = [t1[0]] + [t1[1]] + [t2[1]] + [t2[2]]
                if t1 not in tuples_list:
                    tuples_list.append(t1)
            else:  # index of D < index of B in the sentence (ADBC)
                t2 = [t2[0]] + [t2[1]] + [t1[1]] + [t1[2]]
                if t2 not in tuples_list:
                    tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t2_s == [0, 1]) and (p_t1_s == [1, 2]):
        # Case 3: ABC & BCD
        t1.append(t2[-1])
        if t1 not in tuples_list:
            tuples_list.append(t1)
        if t2 not in tuples_list:
            tuples_list.append(t2)
    elif (p_t1_s == [0, 1]) and (p_t2_s == [1, 2]):
        # Case 3: BCD & ABC
        t2.append(t1[-1])
        if t1 not in tuples_list:
            tuples_list.append(t1)
        if t2 not in tuples_list:
            tuples_list.append(t2)
    elif (p_t1_s == [1, 2]) and (p_t2_s == [0, 2]):
        # Case 4.1: ABC & BDC => ABDC
        pos_t1 = pos[t1[1]][1]  # B
        pos_t2 = pos[t2[1]][1]  # D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        pos_t10 = pos[t1[2]][1]  # C
        pos_t20 = pos[t2[1]][1]  # D
        a1_c0 = in_onto_concepts[w_tk[t1[1]]]
        a2_c0 = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) or (a1_c_filter0 == a2_c_filter0):  # B & D or C & D same concept
            t1 = [t1[0]] + t2  # ABDC
            if t1 not in tuples_list:
                tuples_list.append(t1)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == [0, 2]) and (p_t2_s == [1, 2]):
        # Case 4.2: BDC & ABC => ABDC
        pos_t1 = pos[t1[1]][1]  # B
        pos_t2 = pos[t2[1]][1]  # D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)
        pos_t10 = pos[t1[1]][1]  # D
        pos_t20 = pos[t2[2]][1]  # C
        a1_c0 = in_onto_concepts[w_tk[t1[1]]]
        a2_c0 = in_onto_concepts[w_tk[t2[2]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) | (a1_c_filter0 == a2_c_filter0):  # B & D or C & D same concept
            t2 = [t2[0]] + t1  # ABDC
            if t2 not in tuples_list:
                tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == [0, 1]) and (p_t2_s == [0, 2]):
        # Case 4.3: ABC & ADB => ADBC
        pos_t1 = pos[t1[1]][1]  # B
        pos_t2 = pos[t2[1]][1]  # D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)

        pos_t10 = pos[t1[0]][1]  # A
        pos_t20 = pos[t2[1]][1]  # D
        a1_c0 = in_onto_concepts[w_tk[t1[0]]]
        a2_c0 = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) | (a1_c_filter0 == a2_c_filter0):  # BD or AD same concept
            t2.append(t1[-1])  # ABDC
            if t2 not in tuples_list:
                tuples_list.append(t2)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    elif (p_t1_s == [0, 2]) and (p_t2_s == [0, 1]):
        # Case 4.4: ADB & ABC => ADBC
        pos_t1 = pos[t1[1]][1]  # B
        pos_t2 = pos[t2[1]][1]  # D
        a1_c = in_onto_concepts[w_tk[t1[1]]]
        a2_c = in_onto_concepts[w_tk[t2[1]]]
        a1_c_filter = a1_c
        a2_c_filter = a2_c
        # a1_c_filter = filter_concept(a1_c,pos_t1,concept_assign_dict)
        # a2_c_filter = filter_concept(a2_c,pos_t2,concept_assign_dict)

        pos_t10 = pos[t1[1]][1]  # D
        pos_t20 = pos[t2[0]][1]  # A
        a1_c0 = in_onto_concepts[w_tk[t1[1]]]
        a2_c0 = in_onto_concepts[w_tk[t2[0]]]
        a1_c_filter0 = a1_c0
        a2_c_filter0 = a2_c0
        # a1_c_filter0 = filter_concept(a1_c0,pos_t10,concept_assign_dict)
        # a2_c_filter0 = filter_concept(a2_c0,pos_t20,concept_assign_dict)
        if (a1_c_filter == a2_c_filter) | (a1_c_filter0 == a2_c_filter0):  # BD or AD same concept
            t1.append(t2[-1])  # ABDC
            if t1 not in tuples_list:
                tuples_list.append(t1)
        else:
            if t1 not in tuples_list:
                tuples_list.append(t1)
            if t2 not in tuples_list:
                tuples_list.append(t2)
    return tuples_list


def combine_triple_1(tuples_list, t1, t2, intersect):
    # Only consider case: ABC & CDE => ABCDE
    p_t1_s = [i3 for i3 in range(len(t1)) if t1[i3] in intersect]
    p_t2_s = [i3 for i3 in range(len(t1)) if t2[i3] in intersect]
    if (p_t1_s == 2) and (p_t2_s == 0):  # ABC & CDE
        t1.append(t2[1:])
        if t1 not in tuples_list:
            tuples_list.append(t1)
    elif (p_t1_s == 0) and (p_t2_s == 2):  # CDE & ABC
        t2.append(t1[1:])
        if t2 not in tuples_list:
            tuples_list.append(t2)
    else:  # Dont merge other cases
        if t1 not in tuples_list:
            tuples_list.append(t1)
        if t2 not in tuples_list:
            tuples_list.append(t2)
    return tuples_list
