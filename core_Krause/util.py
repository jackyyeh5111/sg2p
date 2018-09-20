import numpy as np
import os

def decode_paragraphs(sampled_paragraphs, pred, idx2word, fixed_n_sent=False):
    paragraphs = []
    T_stop = 0.5
    for g_idx, paragraph in enumerate(sampled_paragraphs):

        current_paragraph = []
        for s_idx, sent in enumerate(paragraph):
            
            if fixed_n_sent == False:
                if pred[g_idx][s_idx][0] >= T_stop and s_idx >= 1:
                    break
            
            if s_idx >= 6:
                break

            current_sent = ''
            for word_idx in sent:
                current_sent += idx2word[str(word_idx)] + ' '
            
            current_sent = current_sent.replace('<eos> ', '')
            current_sent = current_sent.replace('<pad> ', '')
            current_sent = current_sent + ' . '
            
            current_paragraph.append(current_sent)
            
        paragraphs.append(current_paragraph)

    return paragraphs

def get_max_pred_words(total_preds, num, idx2word):
    total_max_pred_words = []
    for preds in total_preds:
        max_pred_words_paragraph = []
        for pred in preds:
            max_pred_words = []
            for i in xrange(num):
                idx = np.argmax(pred)
                max_pred_words.append(idx2word[str(idx)])
                pred[idx] = -1

            max_pred_words_paragraph.append( ', '.join(max_pred_words) )

        total_max_pred_words.append(max_pred_words_paragraph)

    return total_max_pred_words

def output_max_pred_words(total_max_pred_words, output_path):
    
    with open(output_path, 'w') as f:
        for paragraph_idx, max_pred_words in enumerate(total_max_pred_words):
            for sent_idx, pred_words in enumerate(max_pred_words):
                sent_start = str(paragraph_idx+1) + '-' + str(sent_idx+1)
                f.write( sent_start + ' ' + pred_words + '\n')
                

def output_paragraphs(paragraphs, output_path):
    with open(output_path, 'w') as f:
        for paragraph in paragraphs:
            f.write(' '.join(paragraph).encode('ascii', 'ignore') + '\n' )





