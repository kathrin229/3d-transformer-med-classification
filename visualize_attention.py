"""
Visualizing attention with rolled attention: https://github.com/yiyixuxu/TimeSformer-rolled-attention
"""

from pathlib import Path
from timesformer.models.vit import *
from timesformer.datasets import utils as utils
from timesformer.config.defaults import get_cfg
from einops import rearrange, repeat, reduce
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import matplotlib.pyplot as plt


def show_mask_on_image(img, mask):
  """
    Combining lung slice and heatmap, using a threshold of 0.8 of the maximum value on the heatmap
    Args:
        img (np.array): the lung slice image
        mask (np.array): the attention mask
    Returns:
        cam (np.array): the combined lung slice image and heatmap
        heatmap_orig (np.array): the heatmap without threshold
        heatmap (np.array): the heatmap with threshold
  """
  img = np.stack((img,)*3, axis=-1)
  mask = np.uint8(255 * mask)
  max = mask.max()
  min = mask.min()
  thresh = max * 0.8
  img = np.uint8(img*255)
  heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
  heatmap_orig = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
  heatmap[mask < thresh] = 0
  cam = heatmap + img
  return cam, heatmap_orig, heatmap

def create_masks(masks_in, np_imgs):
  """
    Stacks masks of all lung slices.
    Args:
        masks_in (np.array): the attention masks
        np_imgs (np.array): the lung slices
    Returns:
        masks (np.array): the combined lung slice images and heatmaps
        heatmap_orig (np.array): the heatmaps without threshold
        heatmap (np.array): the heatmaps with threshold
  """
  masks = []
  heatmaps_orig = []
  heatmaps = []
  for mask, img in zip(masks_in, np_imgs):
    mask= cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask, heatmap_orig, heatmap = show_mask_on_image(img, mask)
    masks.append(mask)
    heatmaps_orig.append(heatmap_orig)
    heatmaps.append(heatmap)
  return masks, heatmaps_orig, heatmaps

def combine_divided_attention(attn_t, attn_s):
  """
    Stacks masks of all lung slices.
    Args:
        attn_t (list): time attentions
        attn_s (list): space attentions
    Returns:
        attn_ts (list): combined space and time attentions
  """
  ## time attention
    # average time attention weights across heads
  attn_t = attn_t.mean(dim = 1)
    # add cls_token to attn_t as an identity matrix since it only attends to itself 
  I = torch.eye(attn_t.size(-1)).unsqueeze(0)
  attn_t = torch.cat([I,attn_t], 0)
    # adding identity matrix to account for skipped connection 
  attn_t = attn_t +  torch.eye(attn_t.size(-1))[None,...]
    # renormalize
  attn_t = attn_t / attn_t.sum(-1)[...,None]

  ## space attention
  # average across heads
  attn_s = attn_s.mean(dim = 1)
  # adding residual and renormalize 
  attn_s = attn_s +  torch.eye(attn_s.size(-1))[None,...]
  attn_s = attn_s / attn_s.sum(-1)[...,None]
  
  ## combine the space and time attention
  attn_ts = einsum('tpk, ktq -> ptkq', attn_s, attn_t)
  
  ## average the cls_token attention across the frames
  # splice out the attention for cls_token
  attn_cls = attn_ts[0,:,:,:]
  # average the cls_token attention and repeat across the frames
  attn_cls_a = attn_cls.mean(dim=0)
  attn_cls_a = repeat(attn_cls_a, 'p t -> j p t', j = 32)
  # add it back
  attn_ts = torch.cat([attn_cls_a.unsqueeze(0),attn_ts[1:,:,:,:]],0)
  return(attn_ts)

class DividedAttentionRollout():
  def __init__(self, model, **kwargs):
    self.model = model
    self.hooks = []

  def get_attn_t(self, module, input, output):
    self.time_attentions.append(output.detach().cpu())
  def get_attn_s(self, module, input, output):
    self.space_attentions.append(output.detach().cpu())

  def remove_hooks(self): 
    for h in self.hooks: h.remove()
    
  def __call__(self, path_to_video, attention='dst'):
    input_tensor = tensor_x_train = torch.Tensor(path_to_video)
    # input_tensor = input_tensor.cuda(non_blocking=True)
    self.model.zero_grad()
    self.time_attentions = []
    self.space_attentions = []
    self.attentions = []
    for name, m in model.named_modules():
      if 'temporal_attn.attn_drop' in name:
        self.hooks.append(m.register_forward_hook(self.get_attn_t))
      elif 'attn.attn_drop' in name:
        self.hooks.append(m.register_forward_hook(self.get_attn_s))
    preds = self.model(input_tensor)
    for h in self.hooks: h.remove()

    if attention in ['dst']:
      for attn_t,attn_s in zip(self.time_attentions, self.space_attentions):
        self.attentions.append(combine_divided_attention(attn_t,attn_s))
      p,t = self.attentions[0].shape[0], self.attentions[0].shape[1]
      result = torch.eye(p*t)
      for attention in self.attentions:
        attention = rearrange(attention, 'p1 t1 p2 t2 -> (p1 t1) (p2 t2)')
        result = torch.matmul(attention, result)
      mask = rearrange(result, '(p1 t1) (p2 t2) -> p1 t1 p2 t2', p1 = p, p2=p)
      mask = mask.mean(dim=1)
      mask = mask[0,1:,:]
      width = int(mask.size(0)**0.5)
      mask = rearrange(mask, '(h w) t -> h w t', w = width).numpy()
      mask = mask / np.max(mask)
      return(mask)

    elif attention in ['sl']:
      for attention in self.space_attentions:
        attention = attention.mean(dim = 1)
        attention = attention +  torch.eye(attention.size(-1))[None,...]
        attention = attention / attention.sum(-1)[...,None]

        attn_cls = attention[0,:,:]
        # average the cls_token attention and repeat across the frames
        attn_cls_a = attn_cls.mean(dim=0)
        attn_cls_a = repeat(attn_cls_a, 't -> j t', j = 20)
        self.attentions.append(attention)

      result = self.attentions[0]
      for i in range(1,12):
        result = torch.matmul(self.attentions[i], result)

      # mask = result.mean(dim=1)
      mask = rearrange(result, 't p k -> p k t')
      mask = mask[0,1:,:]
      mask = rearrange(mask, '(t n) x -> (n x) t', n = 4)
      width = int(mask.size(0)**0.5)
      mask = rearrange(mask, '(h w) t -> h w t', w = width).numpy()
      mask = mask / np.max(mask)
      return(mask)

torch.cuda.empty_cache()
model = torch.load('Model_Files/timesformer_dst_3class_160x128x32.pt',map_location='cpu')

loader_CP_test = np.load('Data/dataset_NCP_test_160x128x32.npz')

cat = 'NCP'
attention = 'sl'

dataset_CP_test = loader_CP_test['arr_0']
dataset_CP_test = dataset_CP_test.reshape(-1, 160, 128, 32)
dataset_CP_test = dataset_CP_test[:, :, :, :, np.newaxis]
dataset_CP_test = dataset_CP_test.reshape(-1, 1, 32, 128, 160)
dataset_CP_test = dataset_CP_test[:, :, :, :, np.newaxis]
dataset_CP_test = dataset_CP_test.reshape(-1, 1, 1, 32, 128, 160)

dataset_new = loader_CP_test['arr_0']
dataset_new = dataset_new.reshape(-1, 32, 128, 160)
scan = 11

att_roll = DividedAttentionRollout(model)
masks = att_roll(dataset_CP_test[scan], attention=attention)

masks, heatmap_orig, heatmap = create_masks(list(rearrange(masks, 'h w t -> t h w')),dataset_new[scan]) 
cv2.imwrite('img/visual_images_'+str(scan)+'_'+cat+'_v.jpg', np.vstack(dataset_new[scan]*255))
cv2.imwrite('img/visual_masks_'+str(scan)+'_'+cat+'_'+attention+'_thresh_08_v.jpg', np.vstack(masks))
cv2.imwrite('img/visual_heatmap_orig_'+str(scan)+'_'+cat+'_'+attention+'_thresh_08_v.jpg', np.vstack(heatmap_orig))

print("done")