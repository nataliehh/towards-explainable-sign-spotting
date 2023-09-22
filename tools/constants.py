# Whether to use mouthings or not ()
USE_MOUTHINGS = False

# Fixed length duration of annotations
ANN_LENGTH = 10

# Linguistically relevant columns of SignBank dataset, except for the first two (which are ID columns)
SB_LING_COLS = ["Signbank ID", "Annotation ID Gloss (Dutch)", "Handedness", "Strong Hand", "Weak Hand", 
                "Handshape Change", "Relation between Articulators", "Location", "Relative Orientation: Movement",
                "Relative Orientation: Location", "Orientation Change", "Contact Type", "Movement Shape", 
                "Movement Direction", "Repeated Movement", "Alternating Movement"]

MODEL_PARAMS = {
    'temperature': 0.07,
    'batch_size':  128, # 1024 for the masking experiments! 
    'window_size': 4,
    'learning_rate': 0.001,
    'bidirectional': True,
    'masking': False,
    'dense_size': 256,
    'num_dense': 2,
    'lstm_size': 128,
    'num_lstm': 1,
    'batch_normalisation': True,
    'dropout': 0,
    'lstm_dropout': False,
}

##################################################################
######################### PATH CONSTANTS #########################
##################################################################

# You can adapt these paths in case you change any names, locations
root = './' # root
cngt_data_root  =           root + 'CNGT_data/' 
dataset_root =              root + 'CNGT_datasets/' # contains the features for each CNGT video
np_lmrks_root   =           root + 'CNGT_np_landmarks/' #'CNGT_landmarks_HD/'
cngt_vid_root   =           root + 'CNGT_isolated_signers/' # contains both videos and annotations of CNGT
checkpoint_root =           root + 'model_checkpoints/' # contains the checkpoints of (trained) models

PATHS = {
# Folders
'root': root,
'cngt_data_root':           cngt_data_root,
'cngt_vids_and_eaf':        cngt_vid_root, 
'np_landmarks':             np_lmrks_root, 
'dataset_root':             dataset_root, 
'checkpoint_root':          checkpoint_root,
# Pickle files
'anns_with_tiers':          cngt_data_root + 'anns_with_tiers.pkl',
'cngt_anns' :               cngt_data_root + 'CNGT_annotations.pkl', 
'cngt_demographics':        cngt_data_root + 'cngt_demo.pkl', 
'CNGT_split_ids' :          cngt_data_root + 'CNGT_split_ids.pkl', 
'dataset_anns':             cngt_data_root + 'dataset_anns.pkl', 
'label_encoder':            cngt_data_root + 'label_encoder{}.pkl', 
'man_sim_hand_dist':        cngt_data_root + 'man_sim_hand_dist.pkl', 
'missing_frames_signbank':  cngt_data_root + 'missing_frames_signbank.pkl', 
'signbank_cngt_intersect':  cngt_data_root + 'signbank_intersect.pkl', 
'results_linguistic':       cngt_data_root + 'results_linguistic.pkl',
'results_lmrks':            cngt_data_root + 'results_lmrks.pkl',
'distractors':              cngt_data_root + 'distractors.pkl',
'anns_test':                cngt_data_root + 'anns_test.pkl',
'embeddings':               checkpoint_root+ 'embedding{}.pkl',
'vid_embd_lmrks':           checkpoint_root+ 'vid_embd_lmrks.pkl',
'vid_embd_linguistic':      checkpoint_root+ 'vid_embd_linguistic.pkl',
'confusable_signs_glosses': cngt_data_root + 'confusable_signs_glosses.pkl',
'confusable_signs_times':   cngt_data_root + 'confusable_signs_times.pkl',
'confusable_signs':         cngt_data_root + 'confusable_signs.pkl',
'test_ling_diff_counts':    cngt_data_root + 'test_ling_diff_counts.pkl',
# Numpy files
'data_linguistic':          dataset_root   + 'CNGT_{}_{}{}{}_{}.npy', 
'data_only_lmrks':          dataset_root   + 'CNGT_{}_{}{}{}_{}.npy', 
'features_data':            cngt_data_root + 'features_data{}.npy',
'mirrored_features_data':   cngt_data_root + 'mirrored_features_data{}.npy',  
'masked_corrs':             cngt_data_root + 'masked_corrs{}.npy',  
'out_of_range':             cngt_data_root + 'out_of_range.npy', 
'present_ratios':           cngt_data_root + 'present_ratios.npy', 
'val_vid_ids':              cngt_data_root + 'val_vid_ids.npy', 
'test_vid_ids':             cngt_data_root + 'test_vid_ids.npy', 
'top_signs':                cngt_data_root + 'top{}_signs.npy',
'normalisation':            cngt_data_root + 'normalisation{}{}.npy',
# Csv files
'linguistic_distance_df' :  cngt_data_root + 'ling_dist.csv', 
'linguistic_distance_df_v2':cngt_data_root + 'ling_dist_count_unknowns.csv', 
'signbank_dictionary_info': cngt_data_root + 'signbank_dictionary_info.csv', 
'signbank_handshapes':      cngt_data_root + 'dictionary-export-handshapes.csv', 
'signbank_minimal_pairs':   cngt_data_root + 'dictionary-export-minimalpairs.csv', 
'signbank_with_linguistics':cngt_data_root + 'signbank_with_linguistics.csv', 
# Model checkpoint files
'model_checkpoint':         checkpoint_root+ 'model{}',
}