_base_ = ['/home/lukas/mmpose/configs/body_2d_keypoint/cid/coco/cid_hrnet-w48_8xb20-140e_coco-512x512.py']
test_dataloader = dict(batch_size=1, dataset=dict(data_root='', ann_file='', bbox_file=None, data_prefix=dict(img='')))
test_evaluator = dict(format_only=True)
default_hooks = dict(logger=dict(interval=1))
data_mode = 'bottomup'