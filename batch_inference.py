#! encoding: UTF-8

import os
import glob
import subprocess
from evaluate import evaluate
from optparse import OptionParser


op = OptionParser()
op.add_option("-p",
                  dest="process_name", type=str,
                  help="process name")
op.add_option("-s", dest="start", type=int, default=50)
op.add_option("-e", dest="end", type=int, default=100000 )
op.add_option("--fixed_n_sent",
                  dest="fixed_n_sent", action="store_true", default=False, 
                  help="generate fixed number(S_max) of sent of a paragraph while inferencing")

op.add_option("--rp", dest="repetition_penalty", action="store_true", default=False)
    
(opts, args) = op.parse_args()
if not opts.process_name:  op.error('process_name is not given')

results_path = os.path.join(os.getcwd(), 'results', opts.process_name) # lily
# candidate_paths = glob.glob(results_path + "/val_candidate*")

candidate_paths = []
for path in glob.glob(results_path + "/val_candidate*"):
  if 'coref' not in path:
    candidate_paths.append(path)


# epochs = []
# for candidate_path in candidate_paths:
#   epoch = int(candidate_path.strip('.txt').split('_')[-1])
#   epochs.append(epoch)
# print epochs


candidate_paths = sorted(candidate_paths, key=lambda c_path: int(c_path.strip('.txt').split('_')[-1]))

# for c_path in candidate_paths:
#     epoch = int(c_path.strip('.txt').split('_')[-1])

#     if epoch >= opts.start and epoch <= opts.end:

#         final_scores = evaluate(get_scores=True, candidate_path=c_path)
        
#         msg = ("epoch: %d ==> Bleu_1: %f, Bleu_2: %f, Bleu_3: %f, Bleu_4: %f, METEOR: %f, CIDEr: %f, SPICE: %f" 
#                 % (epoch, final_scores['Bleu_1'], final_scores['Bleu_2'], final_scores['Bleu_3'],
#                 final_scores['Bleu_4'], final_scores['METEOR'], final_scores['CIDEr'], final_scores['SPICE']))

#         print (msg)   


if opts.repetition_penalty:
  o_path = os.path.join(results_path, 'new_score_rpenalty1.5.txt')
else:
  o_path = os.path.join(results_path, 'new_score.txt')

with open(o_path, 'w', buffering=0) as f_score:
    for c_path in candidate_paths:
        epoch = int(c_path.strip('.txt').split('_')[-1])

        if epoch >= opts.start and epoch <= opts.end:

            final_scores = evaluate(get_scores=True, candidate_path=c_path, repetition_penalty=opts.repetition_penalty)
            
            msg = ("epoch: %d ==> Bleu_1: %f, Bleu_2: %f, Bleu_3: %f, Bleu_4: %f, METEOR: %f, CIDEr: %f, SPICE: %f, PRECISION: %f, RECALL: %f" 
                    % (epoch, final_scores['Bleu_1'], final_scores['Bleu_2'], final_scores['Bleu_3'],
                    final_scores['Bleu_4'], final_scores['METEOR'], final_scores['CIDEr'], final_scores['SPICE'], final_scores['SPICE_PRECISION'], final_scores['SPICE_RECALL']))

            print (msg)        
            f_score.write(msg + '\n')

# models = []
# for pretrained_model in pretrained_models:
#   model = pretrained_model.split('/')[-1]
#   model = model.split('.')[0]
#   models.append( model )
#   # print model

# models = list(set(models))
# models = sorted(models, key=lambda x: int(x.split('-')[-1]))
# start_idx=models.index("model-324")
# models = models[start_idx:]


# final_scores = evaluate(get_scores=True, candidate_path=output_path)


# for model in models:
#   arguments = ['-m', 'infer', '--process_name', opts.process_name, '--model_name', model]

#   if opts.fixed_n_sent:
#       arguments += ['--fixed_n_sent']
#   if opts.sentRNN_lstm_dim:
#       arguments += ['--sentRNN_lstm_dim', str(opts.sentRNN_lstm_dim)]
#   if opts.wordRNN_lstm_dim:
#       arguments += ['--wordRNN_lstm_dim', str(opts.wordRNN_lstm_dim)]

#   cmd = 'python incepV3_standard/main.py ' + ' '.join(arguments)

#   os.system( (cmd) )

    # os.system( 'python multilabel_doubleAttn/main.py ' + ' '.join(arguments) )
    # os.system( 'python ' + os.path.join(opts.process_name, "main.py") + ' ' + ' '.join(arguments) )
    

