_base_ = ['./sort_faster-rcnn_fpn_4e_uav-private-half.py']
model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            r"C:\Users\jac41744\mmtracking\uav_ext\work_dirs\faster-rcnn_r50_fpn_4e_uav\new_datset_fasterbest\work_dirs\faster-rcnn_r50_fpn_4e_uav\epoch_5.pth"
            # noqa: E501
        )))
data_root = "C:\\Users\\jac41744\\smartvisiontoolbox\\examples\\data\\uav_cocovid/"
# test_set = 'train'
data = dict(
    train=dict(ann_file=data_root + 'train/coco_train.json',
               img_prefix=data_root + "train"),
    val=dict(ann_file=data_root + 'test/coco_test.json',
             img_prefix=data_root + "test"),
    test=dict(
        ann_file=data_root + f'test/coco_test.json',
        img_prefix=data_root + "test"))
