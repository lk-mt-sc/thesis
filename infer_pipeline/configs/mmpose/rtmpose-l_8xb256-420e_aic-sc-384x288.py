_base_ = ['/home/lukas/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py']
test_dataloader = dict(batch_size=64, dataset=dict(data_root='', ann_file='', bbox_file=None, data_prefix=dict(img='')))
test_evaluator = dict(format_only=True)
default_hooks = dict(logger=dict(interval=1))
data_mode = 'topdown'