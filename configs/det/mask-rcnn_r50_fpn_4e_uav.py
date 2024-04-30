USE_MMDET = True
_base_ = ['./mask-rcnn_r50_fpn_4e_uav-half.py']
# data
data_root = "C:\\Users\\jac41744\\smartvisiontoolbox\\examples\\data\\uav_cocovid/"
data = dict(
    train=dict(ann_file=data_root + 'train/coco_train.json',
               img_prefix=data_root + "train"),
    val=dict(ann_file=data_root + 'test/coco_test.json',
               img_prefix=data_root + "test"),
    test=dict(ann_file=data_root + 'test/coco_test.json',
               img_prefix=data_root + "test"))