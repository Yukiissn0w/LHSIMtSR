import argparse
import os
import sys
import torch
from collections import OrderedDict

from data.segmentation_imp_dataset import SegmentationDataset
from util.visualizer import Visualizer
from util import html
from models.joint_inference_imp_model import JointInference
import util.util as util
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--maskgen_script', type=str,
        default='scripts/test_pretrained_box2mask_city.sh',
        help='path to a test script for box2mask generator')
parser.add_argument('--imggen_script', type=str,
        default='scripts/test_pretrained_mask2image_city.sh',
        help='path to a test script for mask2img generator')
parser.add_argument('--gpu_ids', type=int,
        default=0,
        help='path to a test script for mask2img generator')
parser.add_argument('--how_many', type=int,
        default=50,
        help='number of examples to visualize')
joint_opt = parser.parse_args()

# debug用
joint_opt.gpu_ids = 0
joint_opt.imggen_script = './used_scripts/test_20210130.sh'

joint_opt.gpu_ids = [joint_opt.gpu_ids]
joint_inference_model = JointInference(joint_opt)

# sys.setdefaultencoding("utf-8")

# Hard-coding some parameters
# joint_inference_model.opt_maskgen.load_image = 1
# joint_inference_model.opt_maskgen.min_box_size = 128
min_box_size = 128
# joint_inference_model.opt_maskgen.max_box_size = -1 # not actually used

# opt_maskgen = joint_inference_model.opt_maskgen
opt_pix2pix = joint_inference_model.opt_imggen

# Load data
data_loader = SegmentationDataset()
# data_loader.initialize(opt_maskgen)
data_loader.initialize(opt_pix2pix)

# visualizer = Visualizer(opt_maskgen)
visualizer = Visualizer(opt_pix2pix)

imp_dir = 'test_joint_inference_imp_ori'
# imp_rev_dir = 'test_joint_inference_imp_reverse'

# create website
web_dir = os.path.join('./results', imp_dir, 'val')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s' %
                   ('Joint Inference', 'val'))

# Save directory
save_dir = './results/{}/'.format(imp_dir)
os.makedirs(save_dir, exist_ok=True)

#ch
# print("its: {}".format(len(range(data_loader.dataset_size))))
for i in tqdm(range(data_loader.dataset_size)):
  if i >= joint_opt.how_many:
    break

  # Get data
  raw_inputs, inst_info = data_loader.get_raw_inputs(i)
  img_orig, label_orig = joint_inference_model.normalize_input( \
      raw_inputs['image'], raw_inputs['label'], normalize_image=False)
  # Add a dimension
  img_orig = img_orig.unsqueeze(0)
  label_orig = label_orig.unsqueeze(0)
  # List of bboxes
  bboxs = inst_info['objects'].values()
  #print("bboxs:{}, \n type:{}".format(bboxs, type(bboxs)))

  # Select bbox
  if len(bboxs)==0: continue
  bbox_selected = joint_inference_model.sample_bbox(bboxs, min_box_size)
  #print(bbox_selected)

  #print('generating layout...')
  #   layout, layout_dict, _ = joint_inference_model.gen_layout(
  #           bbox_selected, label_orig, opt_maskgen)
  # layout にはgeneratedではなくGTを使用する
  layout = label_orig

  #print('generating image...')
  imp_label = torch.zeros(1)
  # impression reverse
  imp_label[0] = raw_inputs['imp_label']
  image, test_dict, img_generated = joint_inference_model.gen_image(
          bbox_selected, img_orig, layout, opt_pix2pix, imp_label)
  # imp_label[0] = 1 - raw_inputs['imp_label']
  # image_rev, _, img_generated_rev = joint_inference_model.gen_image(
  #         bbox_selected, img_orig, layout, opt_pix2pix, imp_label)
        
#   visuals = OrderedDict([
#     ('input_image_patch', util.tensor2im(test_dict['image'][0])),
#     # ('predicted_label_patch', util.tensor2label(test_dict['label'][0], opt_maskgen.label_nc)),
#     ('predicted_image_patch', util.tensor2im(img_generated[0])),
#     #('input_mask', util.tensor2label(test_dict['mask_in'][0], 2)),
#     #('label_orig', util.tensor2label(layout_dict['label_orig'][0], opt_maskgen.label_nc)),
#     #('mask_ctx_in_orig', util.tensor2label(layout_dict['mask_ctx_in_orig'][0], opt_maskgen.label_nc)),
#     #('mask_out_orig', util.tensor2im(layout_dict['mask_out_orig'][0])),
#     ('GT_label_canvas', util.tensor2label(label_orig[0], opt_maskgen.label_nc)),
#     ('predicted_label_canvas', util.tensor2label(layout[0], opt_maskgen.label_nc)),
#     ('GT_image_canvas', util.tensor2im(img_orig[0], normalize=False)),
#     ('predicted_image_canvas', util.tensor2im(image[0], normalize=False))
#   ])
  visuals = OrderedDict([
    ('input_image_patch', util.tensor2im(test_dict['image'][0])),
    # ('predicted_label_patch', util.tensor2label(test_dict['label'][0], opt_maskgen.label_nc)),
    ('predicted_image_patch', util.tensor2im(img_generated[0])),
    # ('predicted_image_patch_imprev', util.tensor2im(img_generated_rev[0])),
    #('input_mask', util.tensor2label(test_dict['mask_in'][0], 2)),
    #('label_orig', util.tensor2label(layout_dict['label_orig'][0], opt_maskgen.label_nc)),
    #('mask_ctx_in_orig', util.tensor2label(layout_dict['mask_ctx_in_orig'][0], opt_maskgen.label_nc)),
    #('mask_out_orig', util.tensor2im(layout_dict['mask_out_orig'][0])),
    ('GT_label_canvas', util.tensor2label(label_orig[0], opt_pix2pix.label_nc)),
    # ('predicted_label_canvas', util.tensor2label(layout[0], opt_pix2pix.label_nc)),
    ('GT_image_canvas', util.tensor2im(img_orig[0], normalize=False)),
    ('predicted_image_canvas', util.tensor2im(image[0], normalize=False)),
    # ('predicted_image_canvas_imprev', util.tensor2im(image_rev[0], normalize=False))
  ])
  #print('process image... %s' % ('%05d'% i))
  visualizer.save_images(webpage, visuals, ['%05d' % i])

webpage.save()
