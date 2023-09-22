# imports
import numpy as np
from scipy.spatial.distance import cdist
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm

import importlib

# Keep python tools up to date
from tools import sign_spot_tools
importlib.reload(sign_spot_tools)

# Import all functions from the tools
from tools.sign_spot_tools import*

# Set a numpy seed for consistent results
np.random.seed(123)

# Performs mcnemar's test and prints the results
# We also test the significance here
def mcnemar_test(data, alpha = 0.05, log = True):
    res = mcnemar(data, exact=False)
    chi_stat, p_val = res.statistic, res.pvalue
    significant = p_val < alpha
    symbol = '<' if significant else '>='
    if log:
        print('Chi-square statistic:', chi_stat)
        print('p-value: {} (p {} {})'.format(round(p_val, 5), symbol, alpha))
    return chi_stat, p_val, significant

# Analyze the TP and FN spottings for our model
def get_tp_and_fn_spottings(anns, video_embd_dict, reference_sign_embds, labels, tolerance=25, spotting_thresh=0.5):
    tp, fn = 0, 0
    tp_instances = {} # FP, TN, FN, TN instances with info such as which sign we mistook for the target sign
    total_positives = 0
    for video in tqdm(anns): # Loop over the video_ids
        video_embd = video_embd_dict[video]
        ann = anns[video]
        tp_instances[video] = {}
        
        # Skip signs for which we have no target annotations at all, or no distractors
        str_labels = [sign for sign in ann if len(ann[sign]) > 0 and sign in labels]
        # Get the integer labels given their string values
        int_labels = [labels[sign] for sign in str_labels]
        # Get the reference embeddings of the selected labels 
        select_embds = np.array([reference_sign_embds[l] for l in int_labels])
        # If there are no signs which match the above criteria, we skip the video
        if select_embds.shape[0] == 0:
            continue


        # Compute the distance between each video embedding and each sign embedding
        # We transpose to get the shape (num_signs, num_video_frames)
        distances = cdist(video_embd, select_embds, metric = 'cosine').T
        
        # Loop over the signs and their embeddings
        for i in range(len(str_labels)):
            str_label = str_labels[i]
            sign = labels[str_label] # int label 
            if sign not in reference_sign_embds:
                continue
            if len(str_label) == 0: # If sign is not in train labels, we skip it
                continue
            timestamps = [] # Start with an empty list of timestamps of annotations
            # Get all annotation timestamps of the sign - if they exist
            if str_label in ann: 
                timestamps = sorted(list(set(ann[str_label])))
            # Skip signs for which we have no target annotations at all, or no distractors
            if len(timestamps) == 0:
                continue
            total_positives += len(timestamps)
            # Get tolerance windows for target annotations 
            target_windows = np.array([[t[0]-tolerance, t[0]] for t in timestamps])
            t_w_start, t_w_end = target_windows[:, 0], target_windows[:, 1]
            
            # Compute the distance for each result of using the sliding window
            #dist = cdist(video_embd, reference_embd.reshape(1,-1), metric = 'cosine').flatten()
            dist = distances[i]
            # Get the frames below a threshold cosine distance and make them into timespans
            below_thresh = np.where((dist < spotting_thresh) & (~np.isnan(dist)))[0]
            range_below_thresh = ranges(below_thresh, gap = 2) # Predicted spottings
            # Convert to jump-in-points as the start of the spotting
            jump_in_points = np.array([spot[0] for spot in range_below_thresh])
            spotted_targets = []

            for jip in jump_in_points:
                # A spotted target is one where a JIP falls within the bounds of its tolerance window (start <= JIP <= end)
                spotted_target = np.where(np.logical_and(jip >= t_w_start, jip <= t_w_end))[0]
                if len(spotted_target) > 0:
                    ind = spotted_target[0]
                    target_tuple = (t_w_start[ind], t_w_end[ind]) 
                    if target_tuple not in spotted_targets:
                        spotted_targets.append(target_tuple)

            spotted_targets = list(set(spotted_targets))
            tp += len(spotted_targets)  
            # TPs are targets which are spotted
            tp_instances[video][str_label] = spotted_targets       

    fn = total_positives - tp # FNs: all positives - TPs

    print('\nTP: {:<5}FN: {:<5}'.format(tp, fn))
    return tp_instances, [tp, fn]

def flatten_dict(dict_):
    dict_flattened = []
    for video in dict_:
        vid_dict = dict_[video]
        for sign in vid_dict:
            instances = vid_dict[sign]
            for instance in instances:
                flat = (video, sign,) + instance
                dict_flattened.append(flat)
    return dict_flattened

def get_diff_dicts_at_key(dict1, dict2, key):
    dict1_at_key = []
    dict2_at_key = []
    if key in dict1:
        dict1_at_key = dict1[key]
    if key in dict2:
        dict2_at_key = dict2[key]
    dict1_unique = list(set(dict1_at_key)-set(dict2_at_key))
    dict2_unique = list(set(dict2_at_key)-set(dict1_at_key))
    return dict1_unique, dict2_unique

def print_unique_TPs(unique_TP, sign_counts, print_limit = np.inf):
    print('Unique TP instances per sign')
    # Print which model modes we are comparing (based on which keys are in unique_TP)
    print(' ' * 25,'\t'.join(list(unique_TP.keys())))
    unique_ratios = {}
    i = 0
    for sign in sign_counts:
        total = sign_counts[sign]
        if total == 0:
            continue
        ratio = []
        unique_instances_found = False
        for model_mode in unique_TP:
            unique_TP_mode = unique_TP[model_mode][sign]
            mode_perc = round(unique_TP_mode/total * 100, 1)
            ratio += [mode_perc, unique_TP_mode]
            if mode_perc > 0:
                unique_instances_found = True
        ratio.append(total)
        unique_ratios[sign] = tuple(ratio)
        if i < print_limit and unique_instances_found:
            print('{:<25} ({:<4}%) {:<4}\t({:<4}%) {:<4} \t({:<5} anns)'.format(sign, *ratio))
            i += 1
    return unique_ratios
            
def get_unique_TPs_between_models(mode1, mode2, anns, tp_instances, sign_counts, labels, print_limit = 20, log = True):
    tp_mode1 = tp_instances[mode1]
    tp_mode2 = tp_instances[mode2]
    
    # Instantiate dictionary for keeping tracking of the unique TPs per model mode
    unique_TP = {}
    unique_TP[mode1] = {sign: 0 for sign in labels}
    unique_TP[mode2] = {sign: 0 for sign in labels}
    # For each video, get the TP and FN instances of the landmark feature model and the linguistic feature model
    for i, video in enumerate(anns):
        tp_mode1_vid = tp_mode1[video]
        tp_mode2_vid = tp_mode2[video]

        for sign in anns[video]:
            
            if sign not in unique_TP[mode1]:
                unique_TP[mode1][sign] = 0
            if sign not in unique_TP[mode2]:
                unique_TP[mode2][sign] = 0

            # Get which target instances are only spotted by one model and not the other  
            tp_mode1_unique, tp_mode2_unique = get_diff_dicts_at_key(tp_mode1_vid, tp_mode2_vid, sign)

            # Print which TP and FN instances are unique to each model
            # If one model gets a TP the other doesn't, that automatically means it's a FN
            print_strings = [('TP', mode1), ('TP', mode2)] 

            unique_list = [tp_mode1_unique, tp_mode2_unique]

            print_sign_and_vid = True
            for j in range(len(unique_list)):
                print_string = print_strings[j]
                unique = unique_list[j]
                if len(unique) > 0:
                    # Print the sign, video only the first time
                    if print_sign_and_vid and i == 0 and log:
                        print('Sign: {}, video: {}'.format(sign, video))
                        print_sign_and_vid = False
                    if i == 0 and log:
                        print('{} unique to {} model:'.format(*print_string), unique)
            # Only add a line print if at least one list of TPs was printed
            if not print_sign_and_vid and i == 0 and log:
                print('-'*50)

            unique_TP[mode1][sign] += len(tp_mode1_unique)
            unique_TP[mode2][sign] += len(tp_mode2_unique)
    unique_ratios = print_unique_TPs(unique_TP, sign_counts, print_limit = print_limit)
    return unique_TP, unique_ratios