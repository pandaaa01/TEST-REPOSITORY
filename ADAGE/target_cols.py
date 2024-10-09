from variables import *
def category_column_mapping(premise, candidate_labels):
    '''
    Input: 
        Premise (String containing User Query)
        Candidate Labels (List of strings of possible dataset columns)
    Output:
        List of value pairs containing (Category, Confidence Level)
    '''
    hypothesis = f"This query is connected to {candidate_labels}."
    inputs = zero_shot_tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation_strategy='only_first').to(device)
    logits = zero_shot_model(inputs)[0]
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    entailment_prob = probs[:, 1].item()
    return (candidate_labels, entailment_prob)

def columns_identifier(query):
    target_cols = []
    for i in data.columns:
        entailment_scores = category_column_mapping(query, i)
        if entailment_scores[1] >= 0.90:
            target_cols.append(entailment_scores)

    target_cols.sort(key = lambda x: x[1], reverse= True)
    target_cols = [i[0] for i in target_cols]
    return target_cols
