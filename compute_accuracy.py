from asyncore import write
import torch
from helper_functions import write_to_file


# PRECISION = true_positive / (true_positive + false_positive)
# RECALL = true_positive / (true_positive + false_negative)

def compute_recall_precision(predicted, actual, true, output_file_name):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    class_true_pos = 90*[0]
    class_false_pos = 90*[0]
    class_false_neg = 90*[0]
    
    for i in range(0, len(predicted), 1):
        ### Overall recall and precision
        if (int(predicted[i]) == 0) and (int(actual[i]) == 0):
            true_pos += 1
        elif (int(predicted[i])) == 0 and (int(actual[i]) == 1):
            false_pos += 1
        elif (int(predicted[i])) == 1 and (int(actual[i]) == 0):
            false_neg += 1
            
        ### Class by class recall and precision
        if (int(predicted[i]) == 1) and (int(actual[i]) == 1):
            class_true_pos[true[i] - 1]+=1
        elif (int(predicted[i])) == 1 and (int(actual[i]) == 0):
            class_false_pos[true[i] - 1]+=1
        elif (int(predicted[i])) == 0 and (int(actual[i]) == 1):
            class_false_neg[true[i] - 1]+=1
            
    output = ("overall precision: {}".format(true_pos / (true_pos + false_pos)))
    output += ("overall recall: {}\n".format(true_pos / (true_pos + false_neg)))
    for j in range(0,90,1):
        if class_true_pos[j] > 0:
            output += ("class: {} precision: {}, recall: {}".format(j+1,(class_true_pos[j]/(class_true_pos[j]+class_false_pos[j])), (class_true_pos[j]/(class_true_pos[j]+class_false_neg[j]))))
            output += ("true_pos: {}, false_pos: {}, false_neg: {}".format(class_true_pos[j], class_false_pos[j], class_false_neg[j]))
        else:
            output += ("class: no occurences")
    write_to_file(output, output_file_name=(output_file_name+'_recall_precision'))

def compute_accuracy(model, data_loader, device, test_bool, output_file_name):
    correct_pred, num_examples = 0, 0
    if test_bool:
        predicted_list = []
        actual_list = []
        true_list = []
    for i, (features, targets, true_labels) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        
        #logits, probas = model(features)
        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        if test_bool == True:
            predicted_list.append(predicted_labels)
            actual_list.append(targets)
            true_list.append(true_labels)
    if test_bool == True:
        compute_recall_precision(predicted_list, actual_list, true_list, output_file_name)
    return correct_pred.float()/num_examples * 100
