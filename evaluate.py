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
    for scorer,method in scorers:

        if method == "SPICE":
            score,scores = scorer.compute_score(ref_coref,hypo)
        else:
            score,scores = scorer.compute_score(ref,hypo)

        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score


    return final_scores
    

def evaluate(get_scores=False, 
             reference_path="data/reference.txt", 
             reference_coref_path="data/reference_coref.txt", 
             candidate_path="results/fd-512_att/val_candidate_500.txt"):

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
    final_scores = score(ref, ref_coref, cand)
    
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
        print 'SPICE:\t',final_scores['SPICE']
    

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
    
    
    
    
    
    
    
    
    


