import bert_score
import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.translate import meteor
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import (accuracy_score, f1_score)
from tqdm import tqdm
 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

rouge_scr = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
cc = SmoothingFunction()


def compute_classification_scores(ground_truth_class, generated_labels):
    return round(100*accuracy_score(ground_truth_class, generated_labels), 4), \
            round(100*f1_score(ground_truth_class, generated_labels, average='macro'), 4)
    
def compute_generation_scores(ground_truths_generation, generation_output):

    assert len(ground_truths_generation) == len(generation_output), "Size of predictions and references must be the same"
    assert len(generation_output) != 0, print(generation_output)

    # Metrics Initialization
    N = len(ground_truths_generation)
    sum_bleu = 0
    sum_rouge = 0
    sum_meteor = 0
    sum_bertscore = 0

    new_true = []
    new_pred = []

    for idx in tqdm(range(N)):
        ground_truth = ground_truths_generation[idx]
        generated_text = generation_output[idx]

        if ground_truth == "none" and generated_text == "none":
            sum_bleu += 1
            sum_rouge += 1
            sum_meteor += 1
            sum_bertscore += 1
        elif ground_truth != "none" and generated_text != "none":
            # for Bertscore
            new_true.append(ground_truth)
            new_pred.append(generated_text)

            # SBIC dataset
            if isinstance(ground_truth, list):
                tmp_bleu = 0
                tmp_rouge = 0
                tmp_meteor = 0
                for gt in ground_truth:
                    # Bleu-4
                    tmp_bleu = max(tmp_bleu, nltk.translate.bleu_score.corpus_bleu([word_tokenize(gt)], 
                                                                        [word_tokenize(generated_text)], 
                                                                        weights = [1/4, 1/4, 1/4, 1/4], 
                                                                        smoothing_function=cc.method4))
                    # Rouge
                    rouge = rouge_scr.score(gt, generated_text)
                    tmp_rouge = max(tmp_rouge, rouge["rougeL"].fmeasure)

                    # Meteor
                    try:
                        tmp_meteor = max(tmp_meteor, round(meteor([word_tokenize(gt)], word_tokenize(generated_text)), 1))
                    except:
                        tmp_meteor = max(tmp_meteor, 0.25)
                        
                sum_bleu += tmp_bleu
                sum_rouge += tmp_rouge
                sum_meteor += tmp_meteor

            # IHC dataset
            else:
                # Bleu-4
                sum_bleu += nltk.translate.bleu_score.corpus_bleu([word_tokenize(ground_truth)], 
                                                                    [word_tokenize(generated_text)], 
                                                                    weights = [1/4, 1/4, 1/4, 1/4], 
                                                                    smoothing_function=cc.method4)
                # Rouge
                rouge = rouge_scr.score(ground_truth, generated_text)
                sum_rouge += rouge["rougeL"].fmeasure

                # Meteor
                try:
                    sum_meteor += round(meteor([word_tokenize(ground_truth)], word_tokenize(generated_text)), 1)
                except:
                    sum_meteor += 0.25

    with torch.no_grad():
        sum_bertscore += bert_score.score(new_true, new_pred, lang="en")[2].sum().item()

    return round(100*sum_bleu/N, 4), round(100*sum_rouge/N, 4), round(100*sum_meteor/N, 4), round(100*sum_bertscore/N, 4)
