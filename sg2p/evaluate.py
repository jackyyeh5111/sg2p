import cPickle as pickle
import os
import sys
from optparse import OptionParser
# sys.path.append('../coco-caption')
sys.path.append('../../SPICE/coco-caption')

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

def score(ref, ref_coref, hypo, cand_coref):

    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Cider(),"CIDEr"),
        (Spice(),"SPICE")
    ]

    
    final_scores = {}
    all_scores = {}
    for scorer,method in scorers:

        if method == "SPICE":
            score,scores = scorer.compute_score(ref_coref, cand_coref)
            
            final_scores['SPICE_PRECISION'] = sum([s['All']['pr'] for s in scores]) / len(scores)
            final_scores['SPICE_RECALL'] = sum([s['All']['re'] for s in scores]) / len(scores)
            
            all_scores['SPICE_PRECISION'] = [s['All']['pr'] for s in scores]
            all_scores['SPICE_RECALL'] = [s['All']['re'] for s in scores]
            # print scores.keys()
            # print '-'*5
            # print score
            # raw_input()
        else:
            score,scores = scorer.compute_score(ref,hypo)


        if type(score)==list:
            for m, s, ss in zip(method, score, scores):
                final_scores[m] = s
                all_scores[m] = ss
        else:
            final_scores[method] = score
            all_scores[method] = scores


    return final_scores, all_scores
    
def get_num_repetitions(sg_per_sent_path = "../../SPICE/coco-caption/pycocoevalcap/spice/sg_per_sent.txt", 
                       S_max = 6):

    total_rels = []
    n_repetitions = []
    with open(sg_per_sent_path, 'r') as f:
        for i, line in enumerate(f.readlines()):

            # entry level
            if i % (S_max*2) == 0: # *2 means one line for rels, one line for (attr + obj)s

                if i != 0:
                    n_repetitions.append(repetition)

                sg = {}
                repetition = 0

            # sentence level
            if i % 2 == 0:
                new_key = False
                raw_rels = line.strip().split('|')[:-1] # final item is empty string
                
                # ex: truck, park on, side | side, of, street |

                for raw_rel in raw_rels:
                    sub, predicate, obj = raw_rel.split(',')

                    for key in [raw_rel, sub, obj]:
                        key = key.strip()
                        try:
                            if sg[key]:
                                pass
                        except:
                            sg[key] = True
                            new_key = True

            else:
                raw_attrs_objs = line.strip().split('|')[:-1] # final item is empty string
                for raw_attrs_obj in raw_attrs_objs:
                    # ex: truck & two,fedex,
                    obj, attrs = raw_attrs_obj.split('&')
                    attr_obj_list = [attr.strip() + ' ' + obj.strip() for attr in attrs.split(',')[:-1]]

                    for key in attr_obj_list + [obj]:
                        key = key.strip()
                        try:
                            if sg[key]:
                                pass
                        except:
                            sg[key] = True
                            new_key = True

                if new_key == False:
                    repetition += 1

            # print sg
            # print repetition
            # raw_input()

    # print repetition
    n_repetitions.append(repetition)

    # print n_repetitions

    return n_repetitions 

def evaluate(candidate_path,
             get_scores=False,
             reference_path="../data/reference.txt", 
             reference_coref_path="../data/reference_coref.txt",
             repetition_penalty=True):

    # reference_path = os.path.join("data", reference_file)
    # candidate_path = os.path.join("results", "model-500")

    with open(reference_path, 'r') as f:
        raw_ref = f.readlines()

    with open(reference_coref_path, 'r') as f:
        raw_ref_coref = f.readlines()

    with open(candidate_path, 'r') as f:
        raw_cand = f.readlines()

    # s = 'a man is holding a bunch of bananas . there is a woman in a blue shirt standing next to him . there is a large silver bowl on the table in front of him . '
    # c = 'a man is running . a woman in a red shirt behind a horse .'

    # raw_ref = [s.split('.')] 
    # raw_cand = [c]


    # import hickle
    # n_sents = hickle.load('./data/n_test_data_sents.hkl')


    # make dictionary
    ref = {}
    for i, caption in enumerate(raw_ref):
        caption = caption.strip().lower()
        if caption != '':
            ref[i] = [caption]

    cand = {}
    for i, caption in enumerate(raw_cand):
        caption = caption.strip().lower()
        if caption != '':
            # caption = caption.split(' . ')[:n_sents[i]]
            # caption = caption.split(' . ')[:5]
            # caption = ' . '.join(caption)
            cand[i] = [caption]

    ## for SPICE ###
    ref_coref = {}
    for i, caption in enumerate(raw_ref_coref):
        ref_coref[i] = []
        caption = caption.strip().lower()
        for sent in caption.split('.'):
            if sent != '':
                ref_coref[i] += [sent]

    cand_coref = {}
    for i, caption in enumerate(raw_cand):
        cand_coref[i] = []
        caption = caption.strip().lower()
        for sent in caption.split('.'):
            if sent != '':
                cand_coref[i] += [sent]

    # print cand_coref[0]
    

    # print cand
    # print ref
    # raw_input()
    
    # compute bleu score
    
    final_scores, all_scores = score(ref, ref_coref, cand, cand_coref)
    # final_scores, all_scores = score(ref, ref_coref, cand)
    
    if repetition_penalty:
        # n_repetitions = []
        # for i, caption in enumerate(raw_cand):
        #     cand = {}
        #     repetition = 0
        #     caption = caption.strip().lower()
        #     if caption != '':
        #         captions = caption.split(' . ')
                
        #         for j, caption in enumerate(captions):
        #             try:
        #                 if cand[caption]: # check caption whether is a key
        #                     repetition += 1
        #             except:
        #                 cand[caption] = True

        #     n_repetitions.append(repetition)


        # print n_repetitions[0]
        # print raw_cand[0]


        n_repetitions = get_num_repetitions()
        # print n_repetitions[:5]
        # raw_input()

        all_scores['SPICE'] = [s['All']['f'] for s in all_scores['SPICE']]

        for i in range(len(all_scores['SPICE'])):
            all_scores['SPICE'][i] = all_scores['SPICE'][i] - 0.015 * (n_repetitions[i]**2)

        final_scores['SPICE'] = sum(all_scores['SPICE']) / len(all_scores['SPICE'])

    # print all_scores['SPICE'][:5]
    # all_scores['SPICE'] = [s['All']['f'] for s in all_scores['SPICE']]
    # print sum(all_scores['SPICE']) / len(all_scores['SPICE'])
    # print final_scores['SPICE']


    # raw_input()

    if get_scores:
        return final_scores

    else:
        # print out scores
        print 'Bleu_1:\t',final_scores['Bleu_1']  
        print 'Bleu_2:\t',final_scores['Bleu_2']  
        print 'Bleu_3:\t',final_scores['Bleu_3']  
        print 'Bleu_4:\t',final_scores['Bleu_4']  
        print 'METEOR:\t',final_scores['METEOR']  
        # print 'ROUGE_L:',final_scores['ROUGE_L']  
        print 'CIDEr:\t',final_scores['CIDEr']
        print 'SPICE:\t', final_scores['SPICE']
        print 'SPICE_PRECISION:\t',final_scores['SPICE_PRECISION']
        print 'SPICE_RECALL:\t',final_scores['SPICE_RECALL']



def load_opts():
    op = OptionParser()

    op.add_option("-c", dest="candidate_path", type=str)

    (opts, args) = op.parse_args()
    if not opts.candidate_path:  op.error('candidate_path is not given')
    
    return opts

   
def main():
    
    # opts = load_opts()
    final_scores = evaluate()

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    


