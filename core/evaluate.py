import cPickle as pickle
import os
import sys
from optparse import OptionParser
sys.path.append('../../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def score(ref, hypo):

    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Cider(),"CIDEr")
    ]

    
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)

        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score


    return final_scores
    

def evaluate(get_scores=False, reference_path="data/reference.txt", candidate_path="result/candidate.txt"):

    # reference_path = os.path.join("data", reference_file)
    # candidate_path = os.path.join("results", "model-500")

    with open(reference_path, 'r') as f:
        raw_ref = f.readlines()

    with open(candidate_path, 'r') as f:
        raw_cand = f.readlines()

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

    # print cand[0]
    # print ref[0]
    # raw_input()
    
    # compute bleu score
    final_scores = score(ref, cand)
    
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
    

def load_opts():
    op = OptionParser()

    op.add_option("-c", dest="candidate_path", type=str)

    (opts, args) = op.parse_args()
    if not opts.candidate_path:  op.error('candidate_path is not given')
    
    return opts

   
def main():
    
    opts = load_opts()
    final_scores = evaluate(candidate_path=opts.candidate_path)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    


