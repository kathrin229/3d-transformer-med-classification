from pathlib import Path

import torch
from TimeSformer.timesformer.models.vit import TimeSformer

model_file = Path.home()/'3d-transformer-med-classification/TimeSformer_divST_8x32_224_K600-2.pyth'
model_file.exists()

model = TimeSformer(img_size=224, num_classes=600, num_frames=32, attention_type='divided_space_time',  pretrained_model=str(model_file))

dummy_video = torch.randn(2, 1, 32, 224, 224) # (batch x channels x frames x height x width)

pred = model(dummy_video,) # (2, 600)

assert pred.shape == (2,600)

print('done')