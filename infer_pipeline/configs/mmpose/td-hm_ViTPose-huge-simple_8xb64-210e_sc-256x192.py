_base_ = ['/home/lukas/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192.py']
test_dataloader = dict(batch_size=64, dataset=dict(data_root='', ann_file='', bbox_file=None, data_prefix=dict(img='')))
test_evaluator = dict(format_only=True)
data_mode='topdown'