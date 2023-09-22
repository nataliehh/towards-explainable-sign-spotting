# DATASET CREATION
import numpy as np
import os
from tools.tools import ms_to_frame, load_numpy, find_files, store_dict
from tools.mediapipe_tools import normalise_coordinates, get_pixel_coordinates
from tools.constants import PATHS, ANN_LENGTH
from tools.feature_extraction import get_wrist_angle_distance
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

# This gives indices for zero padding
def zero_padding(ann_len, fixed_length, too_short):
    # If the annotation is too short, we want to pad it with zeros
    if too_short:
        frames_short = fixed_length - ann_len
        chosen_ind = frames_short*[-1] # Flag the annotation to be padded with zeros
    # With annotations that are too long, we undersample them
    else:
        div, mod = divmod(ann_len, fixed_length)
        ind_kept = list(range(0, ann_len, div)) # Keep every X-th frame
        # If we still have too many frames after taking every X-th frame, randomly remove some frames
        # Until we reach the target duration
        chosen_ind = sorted(np.random.choice(ind_kept, fixed_length, replace = False))
    return [chosen_ind]

# Gives indices of frames from oversampling
def over_sampling(ann_len, fixed_length, too_short):
    chosen_inds = []
    if too_short:
        div, mod = divmod(fixed_length, ann_len)
        duplicated = np.array(list(range(ann_len)) * div)
        chosen_ind = np.random.choice(ann_len, mod)
        chosen_ind = sorted(np.append(duplicated,chosen_ind))
    else:
        div, mod = divmod(ann_len, fixed_length)
        ind_kept = list(range(0, ann_len, div))
        chosen_ind = sorted(np.random.choice(ind_kept, fixed_length, replace = False))
    chosen_inds.append(chosen_ind)
    return chosen_inds

# Convert all annotations (variable duration) to a fixed length (X frames)
def ann_to_fixed_length(ann_len, fixed_length, zero_pad = False):
    small, precise, large = [False] * 3
    if ann_len != fixed_length:
        too_short = ann_len < fixed_length
        small, large = too_short, (1-too_short)
        if zero_pad: # Use zero padding (padded frames contain only zeros)
            return zero_padding(ann_len, fixed_length, too_short), small, precise, large
        return over_sampling(ann_len, fixed_length, too_short), small, precise, large
        # return list(range(10)), small, precise, large
    else:
        precise = True
        return [np.arange(ann_len)], small, precise, large
           
# This adds all the data in the list and corresponding labels
# To the data matrix X, and label list y
def add_lmrks(X, y, data_list, label_list):
    for i in range(len(data_list)):
        data, label = data_list[i], label_list[i]
        X.append(data)
        y.append(label)
    return X, y

# Make dataset with the annotations
# Fixed_length = the target size of an annotation (we scaled it to this)
def make_dataset(anns_with_tiers, features, mirrored_features, df, val_vids, test_vids, top_signs, id_split, fixed_length = ANN_LENGTH, 
                 zero_pad = False):
    # Creates empty list for train, train (no augmentation), validation and test sets, as well as the glosses
    X_train, X_train_no_mirr, X_val, X_test, y_train, y_train_no_mirr, y_val, y_test, glosses  = [[] for _ in range(9)]
    # Create lists specifically for the train set where we keep variable annotation lengths
    X_train_var_len, y_train_var_len = [], []
    empty, larger, smaller, precise = [0]*4
    ann_lengths = []

    # Here, we analyze how often landmarks for the hands are detected given the annotations
    # Looping over the .eaf files (i.e. the different signer videos)
    for k in tqdm(anns_with_tiers):
        # print('Processing video {}/{}'.format(i+1, len(list(anns_with_tiers.keys()))), end = '\r')

        # Get the glosses of the file
        anns_dict = anns_with_tiers[k]

        # Adding all elements of the annotations to a dictionary
        # In the format: (start_time, end_time, [side], sign_label)
        items = []
        for key in anns_dict:
            gloss = key.replace('__2H', '')
            # Check if it's an English annotation - we map to Dutch based on Signbank for these
            dutch_equivalent = df.loc[df['Annotation ID Gloss (English)']==gloss]['Annotation ID Gloss (Dutch)']
            # Also check if it isn't already Dutch
            already_dutch = df.loc[df['Annotation ID Gloss (Dutch)']==gloss]['Annotation ID Gloss (Dutch)']
            # Check if there's any English entries equal to this annotation, if so we map to Dutch
            if dutch_equivalent.shape[0] > 0:
                dutch_gloss = dutch_equivalent.tolist()[0]
            elif already_dutch.shape[0] > 0:
                dutch_gloss = gloss   
            # We ignore instances where the annotation doesn't exist in Dutch or English in Signbank
            else: 
                continue

            # If we are using a filtering for top signs, we check if our gloss is in that top list
            if dutch_gloss in top_signs or len(top_signs) == 0:
                # Add the key as an element to each tuple of timespans
                tup_with_key = [(v)+(dutch_gloss,) for v in anns_dict[key]]
                items += tup_with_key

        if len(items) == 0: # If no annotations, skip the rest
            continue

        # Format map e.g. 'S085_CNGT2143.eaf' ->'CNGT2143_S085'
        # To match with the id_split key format
        vid_id = '_'.join(k.replace('.eaf', '').split('_')[::-1])
        train_element = vid_id in id_split['Train']

        # Load the normalised landmarks
        feats = features[k]

        if train_element:
            # We use the mirror-augmented features for the train data
            mirror_feats = mirrored_features[k]

        # Looping over the annotation items in the video
        for item in items:

            # Two-handed signs unpack differently, don't include a 'side' element
            if len(item) == 3:
                start, end, key = item
            else:
                start, end, side, key = item

            # Convert ms to frames to be compatible with mediapipe framewise landmarks
            start_frame = ms_to_frame(start)
            end_frame = ms_to_frame(end)

            # Get frames for the given annotation window
            lmrks = feats[start_frame:end_frame+1]

            # If the annotation is completely out of bounds, we make a note of this and then skip
            if lmrks.shape[0] == 0:
                empty += 0
            else:  
                num_lmrks = lmrks.shape[0]
                lmrks_present = np.unique(np.where(~np.isnan(lmrks))[0]) 

                # If it's a train example, get the landmarks of both hands 
                # For the mirrored data too and then put the landmarks in one array
                if train_element:
                    mirror_lmrks = mirror_feats[start_frame:end_frame+1]

                if lmrks_present.shape[0] == 0:
                    empty+=1
                # If at least some the frames have detected landmarks, we continue processing
                if num_lmrks > 0.0: 
                    ann_lengths.append(num_lmrks)
                    # For annotations not matching exactly with the desired fixed length 
                    # We create a few random samples to make sure we have different examples
                    chosen_inds, s, p, l = ann_to_fixed_length(num_lmrks, fixed_length, zero_pad)

                    smaller += s
                    larger += l
                    precise += p
                    glosses.append(key)
                    for i, chosen_ind in enumerate(chosen_inds):
                        if -1 in chosen_ind: # -1 indicates that we want to use padding
                            num_pads = len(chosen_ind)
                            lmrks_select = np.pad(lmrks, ((0, num_pads), (0,0)))
                            if train_element:
                                mirror_lmrks_select = np.pad(mirror_lmrks, ((0, num_pads), (0,0)))
                        # Get the annotation for the selected frames
                        else:
                            lmrks_select  = lmrks[chosen_ind] 
                            if train_element:
                                mirror_lmrks_select = mirror_lmrks[chosen_ind]

                        # Store the train or test example, for train we also add the mirrored example
                        if train_element:
                            X_train, y_train = add_lmrks(X_train, y_train, [lmrks_select, mirror_lmrks_select], [key, key])
                            X_train_var_len, y_train_var_len = add_lmrks(X_train_var_len, y_train_var_len, [lmrks], [key])
                            # Add every other train element to the non-augmented set because we take 2 samples for augmentation
                            if i % 1 == 0: 
                                X_train_no_mirr,y_train_no_mirr = add_lmrks(X_train_no_mirr,y_train_no_mirr,[lmrks_select],[key])
                        elif vid_id in test_vids:
                            X_test, y_test = add_lmrks(X_test, y_test, [lmrks_select], [key])
                        elif vid_id in val_vids:
                            X_val, y_val = add_lmrks(X_val, y_val, [lmrks_select], [key])
    # Group together the X data, y labels, the annotation length stats and exact lengths
    result = ([X_train, X_train_no_mirr, X_val, X_test], [y_train, y_train_no_mirr, y_val, y_test], 
              [empty, smaller, precise, larger], [X_train_var_len, y_train_var_len], ann_lengths, glosses)
    return result

# Print the lengths of the annotations and some statistics about them
def print_ann_length_stats(ann_lengths, stats):
    empty, smaller, precise, larger = stats

    # Showing how long the annotations are (in frames)
    plt.figure(figsize=(6,3))
    plt.title('Cumulative density plot of annotation length')
    plt.xlim(0,50)
    plt.hist(ann_lengths, bins = 250, cumulative = True, density = True)
    plt.xlabel('Length (in frames)')
    plt.ylabel('Density')
    plt.show()

    print('Mean annotation length', np.median(ann_lengths))
    print('\nNumber of empty anns:', empty)
    total = smaller + larger + precise
    print('Number of annotations:', total)
    smaller_perc =  round(smaller*100/total, 2)
    precise_perc =  round(precise*100/total, 2)
    larger_perc =  round(larger*100/total, 2)
    print('{1}% <= {0} frames, {2}%>= than {0} frames, {3}% = {0} frames'.format(ANN_LENGTH, smaller_perc, precise_perc, larger_perc))

# Custom normalisation method because scipy can cause issues with NaNs
def norm(arr, mean = None, std = None):
    shape = arr.shape
    arr = arr.reshape(-1, shape[-1])
    if mean is None or std is None:
        mean, std = np.mean(arr, axis = 0), np.std(arr, axis = 0)
        std[std == 0] = 1 # Avoid division by zero
    arr_norm = (arr-mean)/std
    arr_norm = arr_norm.reshape(shape)
    return arr_norm, mean, std


# Extract the features, either linguistically motivated ones or just the landmarks
def extract_features(anns_with_tiers, linguistic_features,  mouthing_ind = [13,14]):
    # Path where the extracted features are stored
    features_str = '_only_ling' if linguistic_features else '_only_lmrks'
    features_path = PATHS['features_data'].format(features_str)
    mirrored_path = PATHS['mirrored_features_data'].format(features_str)

    # We find all of the landmark files, split up by the different areas
    # E.g. 'face' has its own numpy file for each video
    path = PATHS['np_landmarks']
    cngt_lmrks = find_files(path, '.npy')
    print('Number of CNGT files when split into numpy files:', len(cngt_lmrks))

    # If the extracted features are not saved yet, we create them and then store them
    if not os.path.exists(features_path) or not os.path.exists(mirrored_path):
        print('Creating feature-extracted data...')
        features_data = {}
        mirrored_data = {}
        for video in tqdm(anns_with_tiers):
            # print('Extracting features for video {}/{}'.format(i+1, len(list(anns_with_tiers.keys()))), end = '\r')
        
            # Loading the numpy files for a specific video
            lmrk_dict = load_numpy(cngt_lmrks, video.replace('.eaf', ''))
            # Normalize the coordinates, also create a horizontally flipped version
            lmrk_dict = normalise_coordinates(get_pixel_coordinates(lmrk_dict))
            pose, face = lmrk_dict['pose'], lmrk_dict['face']
            
            lmrk_dict['l_hand'][:, [0,4,5,17], :] = pose[:, [15,17,19,21], :]
            lmrk_dict['r_hand'][:, [0,4,5,17], :] = pose[:, [16,18,20,22], :]
            
            # Get the landmarks of the upper and lower lip
            mouth = face[:, mouthing_ind, :]
            mouth = mouth.reshape(mouth.shape[0], -1)
            
            if 'only_lmrks' in features_path:
                l_lmrk = lmrk_dict['l_hand']
                r_lmrk = lmrk_dict['r_hand']
                shape = l_lmrk.shape
                l_feats = l_lmrk.reshape(shape[0], -1).astype(np.float32)
                r_feats = r_lmrk.reshape(shape[0], -1).astype(np.float32)
                # For the landmark features, we do not want to engineer extra features
                # So we make a dummy for the distance between the wrists
                distance_wrists = np.array([])
                
                # If we use landmarks instead of linguistic features, we can mirror all x-coordinates
                mirror_ind = np.array(range(0, l_feats.shape[-1], 2))
            else:
                # Getting landmarks preprocessed, with new format being (wrist, distances, angles) for each frame
                l_feats = get_wrist_angle_distance(lmrk_dict, 'l_hand', pose).astype(np.float32)
                r_feats = get_wrist_angle_distance(lmrk_dict, 'r_hand', pose).astype(np.float32)
                distance_wrists = np.abs(r_feats[:,[44,45]] - l_feats[:,[44,45]])
                
                # Only the wrist and fingertip x-coordinates are mirrored with the linguistic features
                mirror_ind = [44] + list(range(46,56,2))
                
            # Concatenate the features of both hands
            features = np.append(l_feats, r_feats, axis = 1)
            if distance_wrists.shape[0] > 0: # Only add wrist distance for linguistic features
                features = np.append(features, distance_wrists, axis = 1)
            # Adding mouthing features
            features = np.append(features, mouth, axis = 1)
            features_data[video] = features
            
            # Store the mirrored data
            l_feats[:,mirror_ind]*= -1 # Mirror all x-coordinates in the data
            r_feats[:,mirror_ind]*= -1
            # We add the features in the reverse order from before (right&left instead of left&right)
            # This is done to mirror the hands
            mirrored_features = np.append(r_feats, l_feats, axis = 1)
            if distance_wrists.shape[0] > 0: # Only add wrist distance for linguistic features
                mirrored_features = np.append(mirrored_features, distance_wrists, axis = 1)
            mirrored_features = np.append(mirrored_features, mouth, axis = 1)
            mirrored_data[video] = mirrored_features
        
        # Store the extracted features in a numpy file (faster than pickling)
        features_data_np = np.array(list(features_data.items()), dtype=object)
        np.save(features_path, features_data_np)
        # Same for the mirrored data
        mirrored_data_np = np.array(list(mirrored_data.items()), dtype=object)
        np.save(mirrored_path, mirrored_data_np)        
    else:
        print('Feature-extracted data exists, loading...')
        # Allow pickle has to be true because we have nested dictionaries
        features_data = dict(np.load(features_path, allow_pickle = True))
        mirrored_data = dict(np.load(mirrored_path, allow_pickle = True))
    print('Done!')
    return features_data, mirrored_data

# Encode the labels that are in the train set
def encode_labels(X, y, top_signs):
    # We want to only keep validation (X index 2) and test (X index 3) examples
    # Which have labels that have been seen during training (X index 0) (i.e. are in the train set)
    in_train_labels = np.unique(y[0])
    val_ind = np.where(np.isin(y[2], in_train_labels))[0]
    test_ind = np.where(np.isin(y[3], in_train_labels))[0]
    X[2], y[2] = np.array(X[2])[val_ind], np.array(y[2])[val_ind]
    X[3], y[3] = np.array(X[3])[test_ind], np.array(y[3])[test_ind]
    print(y[2].shape, y[3].shape)

    top_suffix = '_top' if len(top_signs) != 0 else ''

    # Get the list of possible glosses
    glosses = list(set(y[0]))
    print('Unique glosses:', len(glosses))

    # https://stackoverflow.com/questions/42320834/sklearn-changing-string-class-label-to-int
    # Encode the glosses as numerical
    le = preprocessing.LabelEncoder()
    fitted = le.fit(glosses)

    # Store the transformations of the glosses in a dictionary
    # I.e. string label -> int label
    keys = le.classes_
    values = le.transform(le.classes_)
    dictionary = dict(zip(keys, values))
    store_dict(PATHS['label_encoder'].format(top_suffix), dictionary)
    return fitted, X, y

# Preparing X by removing NaNs, (optionally) normalizing and converting to smaller representation
def prep_X(X, normalize, mean, std):
    X = np.array(X)
    # Convert NaN to numerical
    X = np.nan_to_num(X)
    # Normalize if normalize = true
    if normalize:
        X, _, _ = norm(X, mean, std)
    # Use float32 representation for data to save space and memory for training
    X = X.astype(np.float16)
    return X

def prepare_data(X, y, fitted, normalize, features_str, mouthings_str):
    datasets = ['Train', 'Train (no mirror)', 'Validation', 'Test']
    total_instances = 0
    normalize = True

    # Get the mean & std of the (non-augmented) train data
    X[1] = np.nan_to_num(X[1])
    X_shape = X[1].shape
    X_train_flat = X[1].reshape(-1, X_shape[-1])

    if normalize:
        mean, std = np.mean(X_train_flat, axis = 0), np.std(X_train_flat, axis = 0)
        print('Mean shape:', mean.shape, 'std shape:', std.shape)
    else:
        mean, std = np.zeros(X_shape[-1]), np.ones(X_shape[-1]) 
    # Store the normalisation mean, std for later use
    np.save(PATHS['normalisation'].format(features_str, mouthings_str), [mean, std])

    print('\tdataset \t\t\tdata shape \t\t label shape')
    for i in range(len(X)):
        # Convert labels to numpy
        y[i] = np.array(y[i])
        # Transform the gloss labels (e.g. 'GEBAREN-A') to numerical (e.g. 123)
        y[i] = fitted.transform(y[i])

        # Prepare the data (e.g. normalize, remove NaNs)
        X[i] = prep_X(X[i], normalize, mean, std)

        print('{:<25}\t{:<20}\t\t{}'.format(datasets[i], str(X[i].shape), y[i].shape))
        if datasets[i] != 'Train':
            total_instances += X[i].shape[0]
    return X, y

# Save the datasets and labels to their respective numpy files
def save_dataset(X, y, features_str, mouthings_str, top_suffix):
    # If the features string has '_only_lmrks' in it, then we are using the landmark features
    ling_features = '_only_lmrks' not in features_str
    # Get the train, validation, test data + labels
    X_train, X_train_no_mirr, X_val, X_test = X
    y_train, y_train_no_mirr, y_val, y_test = y
    root = PATHS['data_linguistic'] if ling_features else PATHS['data_only_lmrks']
    # PATHS format: X or y, type of data (train, val, test), features or not, top X signs 
    np.save(root.format('X', 'train', features_str, mouthings_str, top_suffix), X_train)
    np.save(root.format('y', 'train', features_str, mouthings_str, top_suffix), y_train)

    np.save(root.format('X', 'train_no_mirror', features_str, mouthings_str, top_suffix), X_train_no_mirr)
    np.save(root.format('y', 'train_no_mirror', features_str, mouthings_str, top_suffix), y_train_no_mirr)

    np.save(root.format('X', 'val', features_str, mouthings_str, top_suffix), X_val)
    np.save(root.format('y', 'val', features_str, mouthings_str, top_suffix), y_val)

    np.save(root.format('X', 'test', features_str, mouthings_str, top_suffix), X_test)
    np.save(root.format('y', 'test', features_str, mouthings_str, top_suffix), y_test) 
    print('Stored all datasets.')

