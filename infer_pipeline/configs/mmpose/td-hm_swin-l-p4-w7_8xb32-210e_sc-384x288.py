_base_ = ['/home/lukas/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_swin-l-p4-w7_8xb32-210e_coco-384x288.py']
test_dataloader = dict(batch_size=64, dataset=dict(data_root='', ann_file='', bbox_file=None, data_prefix=dict(img='')))
test_evaluator = dict(format_only=True)
data_mode='topdown'