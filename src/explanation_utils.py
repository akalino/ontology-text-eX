from model_utils import normalize


def write_txt(all_explanation, sorted_w_and_i, sorted_norm_w_and_i, target_idx, target_tweet, target_label,
              actual_label, result_path, rules_OLLIE, rules, ollie_anchor_allPos_, anchor_only, onto_only,
              ensemble_ollie_osil_anc, w_LIME):
    with open(result_path, 'a') as outf:
        outf.write('Tweet index: ' + str(target_idx) + '\n\n')
        outf.write('Processed tweet: ' + str(target_tweet) + '\n\n')
        outf.write('Prediction: ' + str(target_label) + ', Actual: ' + str(actual_label) + '\n\n')

        if len(rules_OLLIE) != 0:
            outf.write('OLLIE rules: \n')
            for explain in rules_OLLIE:
                outf.write('   ' + str(explain) + '\n \n')

        if len(onto_only) != 0:
            outf.write('Ontology-based: \n\n')
            for explain in onto_only:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Ontology-based: No explanation found! \n\n')

        if len(anchor_only) != 0:
            outf.write('Anchors: \n\n')
            for explain in anchor_only:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Anchors: No explanation found! \n\n')

        if len(w_LIME) != 0:
            outf.write('LIME: \n\n')
            for explain in w_LIME:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('LIME: No explanation found! \n\n')

        if len(rules) != 0:
            outf.write('Ontology-based - Anchors: \n\n')
            for explain in rules:
                outf.write('   ' + str(explain) + '\n\n')
            outf.write('Normalized: \n\n')
            rules_ = []
            prob = [rules[k][1] for k in range(len(rules))]
            prob_onto = normalize(prob)
            for k in range(len(rules)):
                rules_.append([rules[k][0], prob_onto[k]])
            for explain in rules_:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Ontology-based - Anchors: No explanation found! \n\n')

        if len(ollie_anchor_allPos_) != 0:
            outf.write('OLLIE - Anchors: \n\n')
            for explain in ollie_anchor_allPos_:
                outf.write('   ' + str(explain) + '\n\n')
            outf.write('Normalized: \n\n')
            ollie_anchor_allPos = []
            prob = [ollie_anchor_allPos_[k][1] for k in range(len(ollie_anchor_allPos_))]
            prob_ollie = normalize(prob)
            for k in range(len(ollie_anchor_allPos_)):
                ollie_anchor_allPos.append([ollie_anchor_allPos_[k][0], prob_ollie[k]])
            for explain in ollie_anchor_allPos:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('OLLIE - Anchors: No explanation found! \n\n')

        if len(ensemble_ollie_osil_anc) != 0:
            outf.write('Ensemble results: \n\n')
            for explain in ensemble_ollie_osil_anc:
                outf.write('   ' + str(explain) + '\n\n')
            outf.write('Normalized: \n\n')
            ensemble_ollie_osil_anc_ = []
            prob = [ensemble_ollie_osil_anc[k][1] for k in range(len(ensemble_ollie_osil_anc))]
            prob_ensem = normalize(prob)
            for k in range(len(ensemble_ollie_osil_anc)):
                ensemble_ollie_osil_anc_.append([ensemble_ollie_osil_anc[k][0], prob_ensem[k]])
            for explain in ensemble_ollie_osil_anc_:
                outf.write('   ' + str(explain) + '\n\n')
        else:
            outf.write('Ensemble: No explanation found! \n\n')

        outf.write('\n ============= ********* =============\n\n\n')
    return rules_, ollie_anchor_allPos, ensemble_ollie_osil_anc_
