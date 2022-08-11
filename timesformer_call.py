import yaml
from munch import munchify
from tools.train_net import train



config = yaml.safe_load("""
TRAIN:
    ENABLE: True
    DATASET: kinetics
    BATCH_SIZE: 8
    EVAL_PERIOD: 5
    CHECKPOINT_PERIOD: 5
    AUTO_RESUME: True
    FINETUNE: False
    CHECKPOINT_FILE_PATH: ""
DATA:
    PATH_TO_DATA_DIR: /path/to/kinetics/
    NUM_FRAMES: 8
    SAMPLING_RATE: 32
    TRAIN_JITTER_SCALES: [256, 320]
    TRAIN_CROP_SIZE: 224
    TEST_CROP_SIZE: 224
    INPUT_CHANNEL_NUM: [1]
    MULTI_LABEL: True
TIMESFORMER:
    ATTENTION_TYPE: 'divided_space_time'
SOLVER:
    BASE_LR: 0.005
    LR_POLICY: steps_with_relative_lrs
    STEPS: [0, 11, 14]
    LRS: [1, 0.1, 0.01]
    MAX_EPOCH: 15
    MOMENTUM: 0.9
    WEIGHT_DECAY: 0.0001
    OPTIMIZING_METHOD: sgd
    WARMUP_EPOCHS: 1
    WARMUP_START_LR: 0.005
MODEL:
    MODEL_NAME: vit_base_patch16_224_covid
    NUM_CLASSES: 3
    ARCH: vit
    LOSS_FUNC: cross_entropy
    DROPOUT_RATE: 0.5
TEST:
    ENABLE: True
    DATASET: kinetics
    BATCH_SIZE: 8
    NUM_ENSEMBLE_VIEWS: 1
    NUM_SPATIAL_CROPS: 3
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
    WEIGHT_DECAY: 0.0001
    USE_PRECISE_STATS: False
DETECTION:
    ENABLE: False
LOG_PERIOD: 1
TENSORBOARD:
    ENABLE: False
GLOBAL_BATCH_SIZE: 8
MIXUP:
    ENABLED: False
""")

mymunch = munchify(config)
train(mymunch)

# cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
# num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size
# LOG_PERIOD

print("done")
