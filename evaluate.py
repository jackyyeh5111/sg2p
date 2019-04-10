import cPickle as pickle
import os
import sys
from optparse import OptionParser
# sys.path.append('../coco-caption')
sys.path.append('../SPICE/coco-caption')

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

def score(ref, ref_coref, hypo):

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
            score,scores = scorer.compute_score(ref_coref, hypo)
            
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
    

def evaluate(get_scores=False, 
             reference_path="data/reference.txt", 
             reference_coref_path="data/reference_coref.txt", 
             candidate_path="results/fd-512_att/val_candidate_500.txt",
             repetition_penalty=False):

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




    # print cand
    # print ref
    # raw_input()
    
    # compute bleu score
    final_scores, all_scores = score(ref, ref_coref, cand)
    # final_scores = score(ref, ref_coref, cand)
    
    if repetition_penalty:
        n_repetitions = []
        for i, caption in enumerate(raw_cand):
            cand = {}
            repetition = 0
            caption = caption.strip().lower()
            if caption != '':
                captions = caption.split(' . ')
                
                for j, caption in enumerate(captions):
                    try:
                        if cand[caption]: # check caption whether is a key
                            repetition += 1
                    except:
                        cand[caption] = True

            n_repetitions.append(repetition)


        # print n_repetitions[0]
        # print raw_cand[0]

        all_scores['SPICE'] = [s['All']['f'] for s in all_scores['SPICE']]

        for i in range(len(all_scores['SPICE'])):
            all_scores['SPICE'][i] = all_scores['SPICE'][i] - 0.015 * n_repetitions[i]

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
    
    
    
    
    
    
    
    
    


