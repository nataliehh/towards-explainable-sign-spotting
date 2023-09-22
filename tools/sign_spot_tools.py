import numpy as np
import pandas as pd
import os
import copy
import pympi
from scipy.spatial.distance import cdist
from tools.create_model import get_loss
from tools.make_dataset import norm
from tools.constants import PATHS, ANN_LENGTH, MODEL_PARAMS # path constants
from tools.tools import is_overlap, get_gloss_vals, man_sim_and_hand_dist, load_dict, annotations_ms_to_frame
import matplotlib.pyplot as plt
from tqdm import tqdm

from multiprocessing import Pool

import tensorflow as tf
from keras.optimizers import Adam

np.random.seed(123) # Set random seed for consistency

# The set of features (by index) to remove if we're using linguistic/combined features
# WARNING: you can only remove features that belong to one hand this way (e.g. can't remove the distance between both wrists here)
# If you want to remove two-handed features, see e.g. to_remove_other
to_remove_linguistic = [8, 25, 46, 48, 50, 52, 54, 56, 57]
to_remove_combined = list(set(list(range(44,56))))
double_handed_features = [118,119] # List of features which use both hands

# Compute which features should be removed, given the number of features and if the to-remove indices are for a single hand
# to_remove_other can be used for indices which are not of a specific hand (e.g. features that use both hands)
def compute_to_remove(to_remove_ind, num_features, single_handed, to_remove_other = [118]): #[-2]
    # If we already provided 
    if not single_handed:
       return to_remove_ind + to_remove_other
    # The column indices are 0, 1, ..., num_features
    cols = range(num_features)
    to_remove_other = [cols[t] for t in to_remove_other]
    # print('Removing other feature:', to_remove_other)
    # If we have single handed indices, the provided indices are for the left hand
    to_remove_left = np.array(to_remove_ind)
    # The features from num_one_handed_features/2 onwards are of the right hand (to_remove_right mod num_one_handed_features/2 = to_remove_left) 
    num_one_handed_features = num_features - len(double_handed_features) 
    to_remove_right = to_remove_left + int(num_one_handed_features/2)
    # Compute which columns should remain after removing the specified feature indices
    remain = set(cols) - set(to_remove_left) - set(to_remove_right) - set(to_remove_other)
    return remain

# https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
# Converts consecutive numbers to ranges, e.g. [1,2,3,5,6,8] -> [(1,3), (5,6), (8,8)]
# If we specify gap > 0, then we also allow there to be one or more values missing consecutively
def ranges(nums, gap = 0):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1+abs(gap) < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# Given a range of (start,end) points, compute what ranges lie in between
# E.g. ranges = [(1,2), (5,6)] -> in-between = (2,5)
def between_ranges(ranges, length):
    between = []
    if len(ranges) == 0:
        return [(0,length)]
    # Append a range from the beginning of the video: (0, ...)
    if ranges[0][0] > 0:
        between.append((0, ranges[0][0]))
    for i in range(len(ranges)-1):
        start, end = ranges[i][-1], ranges[i+1][0]
        if start < end:
            between.append((start,end))
    # Append a range for the end of the video (..., end)
    if ranges[-1][-1] < length:
        between.append((ranges[-1][-1], length))
    return between

# Custom z-score method because scipy can cause issues with NaNs
def z_score(arr, mean = None, std = None):
    if mean is None or std is None:
        mean, std = np.mean(arr, axis = 0), np.std(arr, axis = 0)
        std[std == 0] = 1 # Avoid division by zero
    return (arr-mean)/std, mean, std

# You need to use a path with a formatting option here (e.g. containing {})
def load_data(format_path, data_split, to_remove = [], log = False):
    if log: print('Loading {} data...'.format(data_split))
    X_path = format_path.format('X', data_split)
    y_path = format_path.format('y', data_split)
    if log: print('X being loaded...')
    X = np.load(X_path)
    if log: print('original shape:', X.shape)
    
    # Remove NaNs from the data
    if log: print('NaN to num...')
    X = np.nan_to_num(X)
        
    # We remove columns that do not contribute to model performance (based on masking tests)
    if not 'only_lmrks' in X_path:
        remain = compute_to_remove(to_remove, X.shape[-1], single_handed = True)
        X = X[:, :, list(remain)]
        
    if log: print('Converting to smaller float representation...')
    # Train data has to be float32 to fit into memory during training
    X = X.astype(np.float32)
    y = np.load(y_path)
    if log: print('Done.')
    return X, y

def get_data(model_mode, mouthings, log = False):
    # Path to the data and labels
    root = PATHS['dataset_root']
    mouthing_str = '_with_mouthings' if mouthings else ''
    suffix = model_mode + mouthing_str + '__top.npy'
    path = root + 'CNGT_{}_{}' + suffix  
    top = '_top' if 'top' in path else ''

    # Indices of features to remove bc they're highly correlated with other features
    # If we are using only landmarks, we do not remove any features because this removes too many
    to_remove = [] if model_mode == '_only_lmrks' else copy.deepcopy(to_remove_linguistic)
    if model_mode == '_combined':
        to_remove += to_remove_combined

    # Loading train, val, test data and printing their shapes
    data = []
    shapes = []
    splits = ['train', 'train_no_mirror', 'val', 'test']
    for split in splits:
        X, y = load_data(path, split, to_remove = to_remove, log = log)
        data += [X, y]
        shapes.append([str(X.shape), str(y.shape)])
        # print('{} shape: {}\tlabel shape: {}'.format(split, X.shape, y.shape))
    
    display(pd.DataFrame(shapes, columns = ['features shape', 'labels shape'],index = splits).T)
    # Also keep track of the 'top' suffix
    data.append(top)
    return data

# Training the model with data shuffling
def train_model(model, train_batch_gen, val_batch_gen, X_train, y_train, X_val, y_val, labels, target_num_label, batch_size, 
                learning_rate, temperature, num_epochs = 20, plot_freq = 1, patience = 5, decay_rate = 5, ws = 8, num_classes =1000, 
                scce = False):
    model.compile(optimizer=Adam(learning_rate), loss=get_loss(temperature, num_classes, scce=scce))
    # Initialize lists to store history of train, val losses
    val_losses, train_losses = [[] for _ in range(2)]
    increases = 0

    for i in range(num_epochs):
        # Decay the learning rate every X epochs
        if i % decay_rate == 0 and i > 0:
            learning_rate /= 2 # halve the learning rate

            # Recompile model with lower learning rate
            model.compile(optimizer=Adam(learning_rate), loss=get_loss(temperature, num_classes, scce = scce))
            
        print('Epoch {}'.format(i+1))
        history = model.fit(x=train_batch_gen, validation_data = val_batch_gen, epochs=1, verbose = 1)
        val_losses.append(history.history['val_loss'][0])
        train_losses.append(history.history['loss'][0])
        # If the loss increased for X consecutive epochs, we stop
        if len(val_losses) > 1:
            if val_losses[-1] > val_losses[-2]: 
                increases +=1
            else: # If it's not consecutively increasing, reset the counter
                increases = 0
            if increases >= patience: break # Increasing for more than X epochs
        # Can't directly shuffle the generator, so we delete it and make a new one
        del train_batch_gen
        # Shuffle the training data using indices
        train_shuffle = np.random.choice(np.arange(y_train.shape[0]), y_train.shape[0], replace = False)
        X_train, y_train = X_train[train_shuffle], y_train[train_shuffle]
        # Make new batch generator from the newly shuffled data
        train_batch_gen = positive_pairs_batch_gen(X_train, y_train, batch_size = batch_size, window_size = ws)
        # Plot about 1/4th of the validation data
        if i % plot_freq == 0: # and not scce
            plot_sim_of_sampled_embds(model, X_val, y_val, target_num_label, labels, num_samples = 2000, 
                                      log = False, set_str = 'val')
    return val_losses, train_losses

# This makes one reference embedding per sign seen during training, because there are multiple instances of each sign
# Top_rate = the ratio of signs to use (e.g. 0.1 is 10%)
def ref_embds(model, X_train_no_mirr, y_train_no_mirr, y_train_labels, top_ratio = 0.1):
    # Make reference embeddings for each sign based on the train set embeddings
    reference_sign_embds = {}
    all_train_embds = model.predict(X_train_no_mirr, verbose = 0)
    for label in tqdm(y_train_labels):
        # print('embedding sign {}/{}'.format(i+1,len(y_train_labels)), end = '\r')
        # Get all indices where the num. label matches
        ref_inds = np.where(y_train_no_mirr == label)[0]
        # train_ref = X_train_no_mirr[ref_inds]
        N = ref_inds.shape[0] # The num. of instances with that label
        train_embds = all_train_embds[ref_inds]
        # train_embds = model.predict(train_ref, verbose = 0) # Predict
        # Get cosine distance between all the embeddings
        train_ref_dist = cdist(train_embds, train_embds, metric = 'cosine')
        # Get averaged distance from all other reference embeddings
        train_total_dist = np.sum(train_ref_dist, axis = 1)/N
        # We then get the top X% of most representative (closest) embeddings for the sign
        top = max(1,round(N*top_ratio))
        ind = train_total_dist.argsort()[:top]
        top_similar_embds = train_embds[ind]
        # Average these top X% embeddings, store it as the reference embedding for the sign
        reference_res = np.mean(top_similar_embds, axis = 0)
        reference_sign_embds[label] = reference_res
    return reference_sign_embds

# For each target sign in the test set, we check if the right train embedding (of the same sign) ranks
# If the target's train embedding is in the top-k, we count it as an accurate ranking
def top_k_ranking_acc(reference_sign_embds, X_test_pred, y_test, y_train_labels, k_vals = [20]):
    acc= {}
    for k in k_vals:
        acc[k] = 0
    total = 0

    # Get the embeddings of the reference signs and their labels 
    train_embds = np.array(list(reference_sign_embds.values()))
    train_embds_labels = np.array(list(reference_sign_embds.keys()))

    # Go over all sign labels we saw during training
    for label in tqdm(y_train_labels):
        # print('Getting ranking for retrieved sign {}/{}'.format(i+1, len(y_train_labels)), end = '\r')
        
        # Get the test set annotations that match with the sign label
        test_inds = np.where(y_test == label)[0] 
        test_samples = X_test_pred[test_inds] # Get their embeddings

        # For each test set embedding of the current target sign
        for sample in test_samples:
            total += 1
            sample = np.expand_dims(sample, 0)
            # Get the distance with the reference sign embeddings
            dist = cdist(sample, train_embds).flatten()
            for k in k_vals:
                # Get the top X closest reference embeddings to the test embedding
                ind = dist.argsort()[:k]
                # Get the labels corresponding to the top X reference embeddings
                similar_labels = train_embds_labels[ind]
                if label in similar_labels: acc[k]+=1 # If the real label is in the top X, it's accurately ranked

    for k in k_vals:
        print('Number of target signs ranked in top {}: {}/{}'.format(k, acc[k], total))
        print('accuracy@{}: {}%'.format(k, round(100*acc[k]/total,2))) # or is it recall?

def _apply_predict(data):
    model, batches = data[0], data[1]
    pred = model(batches, training = False)
    return pred

def apply_by_multiprocessing(data):
    pool = Pool(processes=os.cpu_count()-1)
    # result = pool.map(_apply_df, np.array_split(list(data.as_numpy_iterator()), workers))
    result = pool.map(_apply_predict, data)
    pool.close()
    return list(result)

# Make embeddings of each video in the val/test set, by windowing over it
def make_video_embds(model, model_mode, mouthings, select_video = '', test_videos = True, to_mask = [], features_data = {}):
    # We don't remove any landmark features, but there is a selection of ling. features to remove
    to_remove = [] if model_mode == '_only_lmrks' else copy.deepcopy(to_remove_linguistic)
    if model_mode == '_combined':
        to_remove += to_remove_combined

    which_set = 'test' if test_videos else 'val'
    print('Loading annotations and getting {} video ids...'.format(which_set))
    if test_videos:
        vid_ids = np.load(PATHS['test_vid_ids']) # Loading ids of the test videos
    else:
        vid_ids = np.load(PATHS['val_vid_ids']) # Loading ids of the val videos
    dataset_root = PATHS['cngt_vids_and_eaf'] # Root where all annotated .eaf sign files are 
    all_anns_with_tiers = load_dict(PATHS['dataset_anns']) # Get video annotations
    # Filter to only the test/val set annotations
    anns_with_tiers = dict(((vid+'.eaf', all_anns_with_tiers[vid+'.eaf']) for vid in vid_ids))

    # If we selected a specific video, only process that one
    if len(select_video) > 0 and select_video in anns_with_tiers:
        anns_with_tiers = {select_video: anns_with_tiers[select_video]}
    # Path where the data features are stored
    print('Loading {} video features...'.format(which_set))
    mouthings_str = '' if not mouthings else '_with_mouthings'
    if len(features_data) == 0:
        features_path = PATHS['features_data'].format(model_mode)
        features_data = dict(np.load(features_path, allow_pickle = True))
    # If we do not want to use mouthing information, we need to remove some features
    # Except for the combined dataset, which automatically uses no mouthings 
    if not mouthings and model_mode != '_combined':
        for video in features_data:
            mouthing_ind = [13,14]
            features_data[video] = features_data[video][:, :-(2*len(mouthing_ind))]

    norm_vals = np.load(PATHS['normalisation'].format(model_mode, mouthings_str))
    mean, std = norm_vals[0], norm_vals[1]

    # Getting the annotations and embeddings for val/test set videos
    video_embd_dict, anns = {}, {}
    batches = []
    batch_slices = {}
    last_batch_ind = 0
    for k in tqdm(anns_with_tiers):
        if not k.replace('.eaf', '') in vid_ids: # Only do this for val/test videos
            continue
        features = features_data[k]
        # Make sure to convert NaN to zeros
        features = np.nan_to_num(features)
        # Normalize the features in the same way as the train data
        features, _ , _ = norm(features, mean, std)
        # print('features mean std shapes', features.shape, mean.shape, std.shape)

        # Making sliding window
        target_shape = (ANN_LENGTH,) + features.shape[1:] # Sliding window size is determined by train data frame length
        batch = np.lib.stride_tricks.sliding_window_view(features, window_shape = target_shape).reshape((-1,)+target_shape)

        batch_slices[k] = (last_batch_ind, last_batch_ind + batch.shape[0])
        last_batch_ind += batch.shape[0]
        batches.append(batch)
    # Predict all batches in two groups to save computational power
    embds = []
    #print(last_batch_ind, last_batch_ind + batch.shape[0])
    batches = np.concatenate(batches)

# I am perturbing here now instead of permuting!
    if len(to_mask) > 0:
        if 118 in to_mask or 119 in to_mask and len(to_mask) == 1:
            to_mask_total = to_mask
            mask = np.random.normal(0.0, 0.1, size=(batches.shape[0], batches.shape[1], len(to_mask)))
        else:
            to_mask_left = np.array(to_mask)
            std = np.std(batches[:,:, to_mask_left])
            mask = np.random.normal(0.0, 0.1, size=(batches.shape[0], batches.shape[1], len(to_mask)*2))
            
            # The features from num_features/2 on are of the right hand (to_remove_right mod num_features/2 = to_remove_left) 
            to_mask_right = to_mask_left + int((batches.shape[-1]-len(double_handed_features))/2)
            to_mask_total = np.append(to_mask_left, to_mask_right)
        batches[:, :, to_mask_total] = mask    #np.random.permutation(batches[:, :, to_mask_total]) # 
        print('Masking feature indices:', to_mask_total)

    # Remove the specified features
    if model_mode != '_only_lmrks':
        remain = compute_to_remove(to_remove, batches.shape[-1], single_handed = True)
        batches = batches[:,:, list(remain)]

    batch_size = MODEL_PARAMS['batch_size']
    print('Creating embeddings...')
    print('Batch shape:', batches.shape)
    for i in tqdm(range(0, batches.shape[0], batch_size)):
        batch = batches[i:i+batch_size]
        output = model(batch, training=False)
        embds.append(output)

    embds = np.concatenate(embds)
    print('Number of embeds:', embds.shape) 
        
    for k in tqdm(anns_with_tiers):
        if not k.replace('.eaf', '') in vid_ids: # Only do this for val/test videos
            continue
        # Getting the landmarks
        eaf_file = pympi.Elan.Eaf(os.path.join(dataset_root, k))
        # Get the glosses and mouthings of the file
        anns_dict, mouthings_dict = get_gloss_vals(eaf_file)
        anns_dict = man_sim_and_hand_dist(anns_dict, manual_sim = False, two_hand_suffix = False)
 
        # Get the right batch 
        batch_start, batch_end = batch_slices[k]
        embd = embds[batch_start:batch_end]
        video_embd_dict[k] = embd
        anns[k] = anns_dict # Store also the corresponding annotations in a dictionary
        
    # Convert the annotations from millisecond timestamps to frames
    anns = annotations_ms_to_frame(anns)
    print('Done.' + ' ' * 100)
    return anns, video_embd_dict

# This ensures the batches contain positive pairs, based on: 
# https://stackoverflow.com/questions/74760839/how-to-generate-batches-in-keras-for-contrastive-learning-to-ensure-positive-pai
# Makes minibatches, batching same labels together. Then makes larger batches with the minibatches.
def positive_pairs_batch_gen(X, y, cat = False, batch_size = 64, window_size = 8):
    cat = len(y.shape) > 1 # Categorical data is 2d, integer labels are 1D
    # We make sure to convert y to be type int64, this is what group_by_window expects
    data = tf.data.Dataset.from_tensor_slices((X, y.astype(np.int64)))
    data = data.group_by_window( 
        # We use the labels to group into minibatches    
        key_func=lambda _, l: tf.where(l==1)[0][0] if cat else l, # Using label (l) as batching-key    
        reduce_func=lambda _, window: window.batch(window_size),     
        window_size=window_size)

    data = data.shuffle(y.shape[0]) # Shuffle the data
    
    # Unbatch the minibatches and batch again based on batch_size
    batch_generator = data.unbatch()
    batch_generator = batch_generator.batch(batch_size)
    return batch_generator

# Computing cosine values between embeddings for given indices of X
# We split the negative & positive pair distances
def compute_cosines(model, indices, X, y, log = True):
    X_ex, y_ex = X[indices], y[indices]
    res = model.predict(X_ex, verbose = 0)
    if log:
        print('Examples:', X_ex.shape)
        print('Prediction result shape:', res.shape)
    cosines_pos, cosines_neg = [], []
    # Get the cosine distances between the examples
    cosines = cdist(res, res, metric = 'cosine')
    for i in range(len(indices)-1):
        for j in range(i+1, len(indices)): 
            cos = cosines[i,j]
            # If it's a positive pair, we add it to the positive pairs list
            if y_ex[i] == y_ex[j]: # Aka same label
                cosines_pos.append(cos)
            else: # We add negative pairs to the negative pairs list
                cosines_neg.append(cos)
    return cosines_pos, cosines_neg

def plot_sim_of_sampled_embds(model, X, y, target_num_label, labels, num_samples = 500, log = False, set_str = 'test'):
    # We plot some sampled cosine similarities (closer to zero is more similar)
    # For random negative and positive pairs and one for a positive pairs of a common sign
    filtered = np.where(y == target_num_label)[0]  # Common sign
    num_entries = filtered.shape[0]
    indices = range(y.shape[0])

    # Random indices (positive and negative pairs)
    ex_indices = np.random.choice(indices, min(num_samples, indices[-1]), replace = False)
    pos_cosines, neg_cosines = compute_cosines(model, ex_indices, X, y, log)

    # Indices selected from specific, common sign (e.g. GEBAREN-A)
    ex_indices = np.random.choice(filtered, min(num_samples, num_entries), replace = False)
    filtered_pos_cosines, _ = compute_cosines(model, ex_indices, X, y, log)

    # Making lists of the histograms and their titles to make it easier to plot
    plots_cos = [neg_cosines, pos_cosines, filtered_pos_cosines]
    plots_titles = ['Randomly selected negative pairs', 'Randomly selected positive pairs', 
                   'Positive pairs for the sign {}'.format(find_target_label(target_num_label,labels)[0])]
    # Use subplots to plot them next to each other
    fig, ax = plt.subplots(1, 3, figsize=(20,4))
    fig.suptitle('Cosine distance between positive/negative pair embeddings ({} set)'.format(set_str), fontsize = 18)
    
    # Loop over the embedding cosine distances and plot them
    for i in range(len(plots_cos)):
        cos = plots_cos[i]
        title = plots_titles[i]
        negative = 'negative' in title # Whether the plot is of negative pairs
        ax[i].set_xlim(0,1)
        # We assume that each bin will not have a higher count than some fraction of the total number of instances
        # To make the y-axis a consistent length and so we don't make the plots way too tall to see the bins
        div = 3 if negative else 4
        ax[i].set_ylim(0, round(len(cos)/div)) 
        ax[i].hist(cos, bins = 30, color = 'crimson' if negative else 'mediumseagreen')
        ax[i].set_title(title) 
        ax[i].set_xlabel('Cosine distance')
        ax[i].set_ylabel('Counts')
        # We also print the ratio of examples that make the cut below/above 0.5 (for pos pairs: below)
        thresh = 0.5
        less_or_greater = '>' if negative else '<'
        num_dist_thresh = len([c for c in cos if c >= thresh]) if negative else len([c for c in cos if c <= thresh])
        print("Ratio of {} {}= {}: {}".format(title.lower(), less_or_greater, thresh, round(num_dist_thresh/len(cos), 3)))
    plt.tight_layout()
    plt.show()

# We find the string equivalent of a numerical label (e.g. 123 -> 'SIGN-A')
def find_target_label(target_num_label, labels):
    target_label = [x for x in labels.items() if x[1] == target_num_label]
    if len(target_label) != 0:
        target_label = target_label[0]
    else:
        target_label = ('', -1)
    return target_label

# Get the distractors for each video and each target sign, making sure that the distractors
# Do not overlap with each other or with target sign annotations, for a given tolerance window size
def get_distractors(anns, labels, dist_df, tolerance = 75, balanced = False, random = False, forbidden_distractors = []):
    distractor_times, distractor_glosses = {}, {}
    linguistic_distances = []
    for video in (anns): # tqdm
        # print('video {}/{}'.format(i+1, len(video_ids)), end = '\r')
        distractor_times[video] = {}
        distractor_glosses[video] = {}
        ann = anns[video]
        for sign in ann:
            if sign not in labels or sign not in dist_df.columns.tolist():
                continue 
            distractor_times[video][sign] = []
            distractor_glosses[video][sign] = []
                
            # Get all annotation timestamps of the sign - if they exist
            timestamps = sorted(list(set(ann[sign])))
            # Skip signs for which we have no timestamps at all
            if len(timestamps) == 0:
                continue
                
            # We try to balance the number of distractors (should be equal to the number of target anns)
            num_distractors = len(timestamps)
            # Get tolerance windows for target annotations 
            target_starts = [t[0] for t in timestamps]
            
            # Shuffle the similarity randomly if random_distractors = True
            # We set a random state to allow us to reproduce the results
            if random:
                similar_signs = dist_df[sign].sample(frac=1, random_state = 123).keys()
            else:
                # Go over all other signs in order of similarity
                similar_signs = dist_df[sign].sort_values().keys()
            for sim_sign in similar_signs:
                # Make sure the distractor candidate sign is a known train label, occurs in the video
                # And also is not the same as the target sign
                if sim_sign in ann and sim_sign != sign and sim_sign in labels:
                    for a in ann[sim_sign]:
                        # Stop if we have an equal number of distractors to annotations of the target sign
                        if len(distractor_times[video][sign]) == num_distractors:
                            break
                        if [video, sign, str(a[0]), sim_sign] in forbidden_distractors:
                            continue
                        # Get the tolerance window of the candidate distractor
                        candidate_distractor = (a[0]-tolerance, a[0])
                        # Make sure the distractor does not overlap with any other distractor or a target
                        overlap = False
                        # Go over the annotations (distractors or targets) that are already selected
                        # And only keep the new candidate distractor if it doesn't overlap with them
                        for chosen_anns in distractor_times[video][sign] + target_starts:
                            ann_with_tolerance = (chosen_anns-tolerance, chosen_anns)
                            if is_overlap(ann_with_tolerance, candidate_distractor, offset = 0):
                                overlap = True
                                break
                        # Skip distractors which have overlap with another distractor or target,
                        # Or which are in the set of distractors to skip (aka forbidden distractors)
                        if not overlap:
                            distractor_times[video][sign].append(a[0])
                            distractor_glosses[video][sign].append(sim_sign)
                            linguistic_distances.append(dist_df[sign][sim_sign])
                    # If we want a balanced number of distractors (num. targets = num. distractors)
                    # And there's not enough distractors, we reset the distractor list to empty
                    if balanced and len(distractor_times[video][sign]) < num_distractors:
                        distractor_times[video][sign] = []
    return distractor_times, distractor_glosses, linguistic_distances

# This computes the TP, FP, TN, FNs for accuracy, precision and recall
# Tolerance should be provided as frames, not seconds
def distractor_based_eval(anns, video_embd_dict, reference_sign_embds, labels, distractor_times,
                          distractor_glosses, tolerance=25, spotting_thresh=0.5, random=False, log = False):
    tp, fp, tn, fn = [0]*4
    instances = {} # FP, TN, FN, TN instances with info such as which sign we mistook for the target sign
    total_positives = 0
    total_distractors = 0
    for eval in ['TP', 'FN', 'FP', 'TN']:
            instances[eval] = {}
    for video in tqdm(anns, disable = not log): # Loop over the video_ids
        video_embd = video_embd_dict[video]
        ann = anns[video]
        # Skip signs for which we have no target annotations at all, or no distractors
        str_labels = [sign for sign in ann if len(ann[sign]) > 0 and sign in distractor_times[video] and len(distractor_times[video][sign]) > 0
                    and sign in labels]
        # Get the integer labels given their string values
        int_labels = [labels[sign] for sign in str_labels]
        # print('int labels:', len(int_labels))
        # Get the reference embeddings of the selected labels 
        select_embds = np.array([reference_sign_embds[l] for l in int_labels])
        # print('select embds:', select_embds.shape)
        # If there are no signs which match the above criteria, we skip the video
        if select_embds.shape[0] == 0:
            continue

        # Compute the distance between each video embedding and each sign embedding
        # We transpose to get the shape (num_signs, num_video_frames)
        distances = cdist(video_embd, select_embds, metric = 'cosine').T

        for eval in instances:
            instances[eval][video] = {}
        
        # Loop over the signs and their embeddings
        for i in range(len(str_labels)):
            str_label = str_labels[i]
            # Get all annotation timestamps of the sign - if they exist
            timestamps = sorted(list(set(ann[str_label])))

            total_positives += len(timestamps)
            # Get tolerance windows for target annotations 
            target_windows = np.array([[t[0]-tolerance, t[0]] for t in timestamps])
            t_w_start, t_w_end = target_windows[:, 0], target_windows[:, 1]
            if random: # Random baseline uses randomly generated distances
                dist = np.random.rand(video_embd.shape[0])
                #below_thresh = np.where((dist < spotting_thresh) & (~np.isnan(dist)))[0]
            else:
                # Get the distance for each window in the video, compared to the target sign
                # dist = distances[i] #cdist(video_embd, reference_embd.reshape(1,-1), metric = 'cosine').flatten()
                dist = distances[i]
            below_thresh = np.where((dist < spotting_thresh) & (~np.isnan(dist)))[0]

            # Get the frames below a threshold cosine distance and make them into timespans
            range_below_thresh = ranges(below_thresh) # Predicted spottings # gap = 2
            # Convert to jump-in-points as the start of the spotting plus a small constant to allow for overlap
            jump_in_points = np.array([spot[0] for spot in range_below_thresh])
            correct_jump_in_points = [] # Correct jump-in-points (JIPs for short)
            spotted_targets = []
            
            # Loop over the jump-in-points (JIPs)
            for jip in jump_in_points:
                # A spotted target is one where a JIP falls within the bounds of its tolerance window (start <= JIP <= end)
                spotted_target = np.where(np.logical_and(jip >= t_w_start, jip <= t_w_end))[0]
                if len(spotted_target) > 0:
                    ind = spotted_target[0]
                    target_tuple = (t_w_start[ind], t_w_end[ind]) 
                    if target_tuple not in spotted_targets:
                        spotted_targets.append(target_tuple)
                        correct_jump_in_points.append(jip)

            spotted_targets = list(set(spotted_targets))
            correct_jump_in_points = set(correct_jump_in_points)
            tp += len(spotted_targets)  
            # TPs are targets which are spotted, whereas FNs are not-spotted targets
            instances['TP'][video][str_label] = spotted_targets 
            # Convert target_windows lists to tuples to allow for making sets of them 
            target_windows = map(lambda x: tuple(x), target_windows)
            instances['FN'][video][str_label] = list(set(target_windows)-set(spotted_targets))      

            # Get JIPs which have not been matched 
            wrong_jump_in_points = list(set(jump_in_points)-correct_jump_in_points) 
            
            # We then use the distractors of the target sign and keep track of which of them are spotted
            spotted_distractors = []
            distractor_windows = np.array([[d-tolerance, d] for d in distractor_times[video][str_label]])
            d_w_start, d_w_end = distractor_windows[:, 0], distractor_windows[:,1]
            distractor_time_and_gloss = []

            for i in range(d_w_start.shape[0]):
                distractor_tuple = (d_w_start[i], d_w_end[i], distractor_glosses[video][str_label][i])
                distractor_time_and_gloss.append(distractor_tuple)
            # Check which jump-in points that didn't match a target, match with a distractor
            for jip in wrong_jump_in_points:
                spotted_distractor_inds = np.where(np.logical_and(jip >= d_w_start, jip <= d_w_end))[0]
                if len(spotted_distractor_inds) > 0:
                    ind = spotted_distractor_inds[0]
                    t_d = (d_w_start[ind], d_w_end[ind])
                    gloss_d = distractor_glosses[video][str_label][ind]
                    distractor_tuple = t_d + (gloss_d,)
                    if distractor_tuple not in spotted_distractors:
                        spotted_distractors.append(distractor_tuple)
                        
            # Compute the number distractors
            spotted_distractors = list(set(spotted_distractors))
            num_negatives = len(list(set(distractor_times[video][str_label])))
            total_distractors += num_negatives
            instances['FP'][video][str_label] = spotted_distractors
            instances['TN'][video][str_label] = list(set(distractor_time_and_gloss) - set(spotted_distractors))

            fp += len(spotted_distractors) # All non-targets which are spotted are FPs
            
            # Get the TNs for the sign, video: all negatives that are not FPs are TNs
            tn += num_negatives - len(spotted_distractors)

    fn = total_positives - tp # FNs: all positives - TPs

    # Compute metrics (accuracy, precision, recall) and get total classifications
    total = tp+tn+fp+fn
    targets = tp+fn # Total number of target anns
    distractors = fp+tn # Number of distractors
    acc = round((tp+tn)/total, 3)
    precision = round(tp/(tp+fp), 3)
    recall = round(tp/targets, 3)
    f1 = round((2*recall*precision)/(precision+recall),3)
    
    if log:
        print('\nTP: {:<12}FP: {:<12}FN: {:<12}TN: {:<12}'.format(tp, fp, fn, tn))
        print('Accuracy: {}\tPrecision: {}\tRecall: {}\tF1-score:{}'.format(acc, precision, recall, f1))
        print('Total judgments: {} ({} targets, {} distractors)'.format(total, targets, distractors))
        
    return instances, [tp, fn, fp, tn, acc, precision, recall, f1]