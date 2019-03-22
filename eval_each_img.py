import cPickle as pickle
import os
import sys
sys.path.append('../SPICE/coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from optparse import OptionParser
from pycocoevalcap.spice.spice import Spice
import numpy as np
import json

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
    

def evaluate(opts, 
             get_scores=False, 
             reference_path="data/reference.txt", 
             reference_coref_path="data/reference_coref.txt", 
             candidate_path="result/candidate.txt"):

    # reference_path = os.path.join("data", reference_file)
    candidate_path = os.path.join("results", opts.process_name, "val_candidate_%d.txt" % opts.epoch)
    candidate_coref_path = os.path.join("results", opts.process_name, "val_candidate_%d_coref.txt" % opts.epoch)


    with open(reference_path, 'r') as f:
        raw_ref = f.readlines()

    with open(reference_coref_path, 'r') as f:
        raw_ref_coref = f.readlines()

    with open(candidate_path, 'r') as f:
        raw_cand = f.readlines()

    with open(candidate_coref_path, 'r') as f:
        raw_cand_coref = f.readlines()


    # load img_ids
    densecap_test_path = "./data/image_path/imgs_test_path.txt"
    image_paths = open(densecap_test_path).read().splitlines()
    img_ids = np.array(map(lambda x: os.path.basename(x).split('.')[0], image_paths))


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

    cand_coref = {}
    for i, caption in enumerate(raw_cand_coref):
        caption = caption.strip().lower()
        if caption != '':
            cand_coref[i] = [caption]

    # print ref
    # print cand
    # raw_input()

    # compute bleu score
    final_scores, all_scores = score(ref, ref_coref, cand, cand_coref)

    evals = {
        'bleus': {},
        'meteors': {},
        'ciders': {},
        'spice': {},
        'spice_precision': {},
        'spice_recall': {}
    }

    bleus = np.array(map(lambda b1, b2, b3, b4: b1+b2+b3+b4, all_scores['Bleu_1'], all_scores['Bleu_2'] , all_scores['Bleu_3'] , all_scores['Bleu_4'] ))
    bleus_idx = bleus.argsort()

    meteors =np.array(map(lambda m: m, all_scores['METEOR']))
    meteors_idx = meteors.argsort()
    
    ciders = np.array(map(lambda m: m, all_scores['CIDEr']))
    ciders_idx = ciders.argsort()

    spice = np.array(map(lambda m: m, all_scores['SPICE']))
    spice_idx = spice.argsort()

    spice_precision = np.array(map(lambda m: m, all_scores['SPICE_PRECISION']))
    spice_recall = np.array(map(lambda m: m, all_scores['SPICE_RECALL']))
   
    evals["bleus"]['img_ids'] = img_ids[bleus_idx]
    evals["meteors"]['img_ids'] = img_ids[meteors_idx]
    evals["ciders"]['img_ids'] = img_ids[ciders_idx]
    evals["spice"]['img_ids'] = img_ids[spice_idx]

    evals["bleus"]['scores'] = np.sort(bleus)
    evals["meteors"]['scores'] = np.sort(meteors)
    evals["ciders"]['scores'] = np.sort(ciders)
    evals["spice"]['scores'] = np.sort(spice)
    evals["spice_precision"]['scores'] = spice_precision[spice_idx]
    evals["spice_recall"]['scores'] = spice_recall[spice_idx]


    output_path = os.path.join("results", opts.process_name, "eval_each_img_scores_%d.pkl" % opts.epoch)
    with open(output_path, 'w') as f:
        pickle.dump(evals, f)

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
        print 'SPICE_PRECISION:\t',final_scores['SPICE_PRECISION']
        print 'SPICE_RECALL:\t',final_scores['SPICE_RECALL']


    # make dictionary
    # ref = {}
    # for i, caption in enumerate(raw_ref):
    #     caption = caption.strip().lower()
    #     if caption != '':
    #         ref[i] = [caption]

    # cand = {}
    # for i, caption in enumerate(raw_cand):
    #     caption = caption.strip().lower()
    #     if caption != '':
    #         cand[i] = [caption]


 
    # print cand[0]
    # print ref[0]
    # raw_input()
    
    
    
   
def main():

    op = OptionParser()
    op.add_option("--process_name", dest="process_name", type=str)
    op.add_option("--epoch", dest="epoch", type=int)

    (opts, args) = op.parse_args()
    if not opts.process_name:  op.error('process_name is not given')
    if not opts.epoch:  op.error('epoch is not given')
    

    final_scores = evaluate(opts)
    
if __name__ == "__main__":
    main()
    
    
    




    
    
    
    
    


