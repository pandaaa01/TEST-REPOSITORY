import torch
from variables import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def task_identifcation(premise, candidate_labels):
    '''
    Input: 
        Premise (String containing User Query)
        Candidate Labels (List of strings that could be the category of the premise)
    Output:
        List of value pairs containing (Category, Confidence Level)
    '''
    entailment_scores = []
    
    for label in candidate_labels:
        hypothesis = f"This task assigned is {label}."
        inputs = zero_shot_tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation_strategy='only_first').to(device)
        logits = zero_shot_model(inputs)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        entailment_prob = probs[:, 1].item()
        entailment_scores.append((label, entailment_prob))
    
    entailment_scores = sorted(entailment_scores, key=lambda x: x[1], reverse=True)
    return entailment_scores[0][0]

