from tensorflow import keras
from constants import PATHS, USE_MOUTHINGS
from create_model import SupCon_loss
from sign_spot_tools import make_video_embds
import time

print('Setting up...')
model_mode = '_only_ling'
model_path = PATHS['model_checkpoint'].format(model_mode)
model = keras.models.load_model(model_path, custom_objects={'SupCon_loss':SupCon_loss},
                                               compile=False) 

start_t = time.time()
print('Predicting...')
anns, video_embd_dict = make_video_embds(model, model_mode, USE_MOUTHINGS, select_video = '', test_videos = True, to_permute = [])
print('Done. Time:', time.time() - start_t)

