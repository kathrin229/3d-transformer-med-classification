"""
Training and testing TimeSformer models
"""

import yaml
from munch import munchify
from tools.train_net import train
from tools.test_net import test

from data_loading import load_dataset_train_valid, load_dataset_test

import time


## Config oriented at timesformer/config/defaults.py from https://github.com/facebookresearch/TimeSformer
config = yaml.safe_load("""
FRAMEWORK: 'pytorch'
TRAIN:
    ENABLE: True
    DATASET: kinetics
    BATCH_SIZE: 8
    EVAL_PERIOD: 5
    CHECKPOINT_PERIOD: 5
    AUTO_RESUME: False
    FINETUNE: False
    CHECKPOINT_FILE_PATH: ""
    CHECKPOINT_TYPE: "pytorch"
    CHECKPOINT_INFLATE: False
    CHECKPOINT_EPOCH_RESET: False
    CHECKPOINT_CLEAR_NAME_PATTERN: ()
DATA:
    PATH_TO_DATA_DIR: /path/to/kinetics/
    NUM_FRAMES: 8
    SAMPLING_RATE: 32
    TRAIN_JITTER_SCALES: [256, 320]
    TRAIN_CROP_SIZE: 224
    TEST_CROP_SIZE: 224
    INPUT_CHANNEL_NUM: [1]
    MULTI_LABEL: False
    ENSEMBLE_METHOD: "sum"
    CLASSES: ['CP', 'NCP']
    SIZE: '160x128x32'
TIMESFORMER:
    ATTENTION_TYPE: 'time_limited'
    
MODEL:
    PRETRAINED_MODEL_NAME: 'Model_Files/TimeSformer_divST_8x32_224_K600.pyth'
    MODEL_NAME: 'timesformer_tl_2class_160x128x32.pt'
    NUM_CLASSES: 2
    ARCH: vit
    LOSS_FUNC: cross_entropy
    DROPOUT_RATE: 0.5
SOLVER:
    BASE_LR: 0.005
    LR_POLICY: steps_with_relative_lrs
    STEPS: [0, 11, 14]
    LRS: [1, 0.1, 0.01]
    MAX_EPOCH: 20
    MOMENTUM: 0.9
    WEIGHT_DECAY: 0.0001
    OPTIMIZING_METHOD: sgd
    WARMUP_EPOCHS: 0.0
    WARMUP_START_LR: 0.01
    DAMPENING: 0.0
    NESTEROV: True
TEST:
    ENABLE: True
    DATASET: kinetics
    BATCH_SIZE: 8
    NUM_ENSEMBLE_VIEWS: 1
    NUM_SPATIAL_CROPS: 1
    CHECKPOINT_FILE_PATH: ""
    CHECKPOINT_TYPE: "pytorch"
    SAVE_RESULTS_PATH: ""
DATA_LOADER:
    NUM_WORKERS: 8
    PIN_MEMORY: True
NUM_GPUS: 1
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
LOG_MODEL_INFO: False
MULTIGRID:
    LONG_CYCLE: False
    SHORT_CYCLE: False
BN:
    WEIGHT_DECAY: 0.0
    USE_PRECISE_STATS: False
    NUM_BATCHES_PRECISE: 200
    NORM_TYPE: "batchnorm"
    NUM_SPLITS: 1
    NUM_SYNC_DEVICES: 1
DETECTION:
    ENABLE: False
LOG_PERIOD: 10
TENSORBOARD:
    ENABLE: False
    CONFUSION_MATRIX:
        ENABLE: True
        FIGSIZE: [8, 8]
        SUBSET_PATH: ""
    HISTOGRAM:
        ENABLE: True
        FIGSIZE: [8, 8]
        SUBSET_PATH: ""
        TOPK: 2
    LOG_DIR: ""
    PREDICTIONS_PATH: ""
    CLASS_NAMES_PATH: ""
    CATEGORIES_PATH: ""
GLOBAL_BATCH_SIZE: 8
MIXUP:
    ENABLED: False
""")

config_munch = munchify(config)

start_time = time.time()
train(config_munch)

fit_time = time.time()
test(config_munch)

stop_time = time.time()

print("Training: --- %s seconds ---" % (fit_time - start_time))
print("Testing: --- %s seconds ---" % (stop_time - fit_time))
print("Total: --- %s seconds ---" % (stop_time - start_time))

print("done")
