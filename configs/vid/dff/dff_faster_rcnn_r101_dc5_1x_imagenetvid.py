_base_ = ['./dff_faster_rcnn_r50_dc5_1x_imagenetvid.py']
model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='/mnt/lustre/share_data/PAT/datasets/mmtrack/pretrain/resnet101-63fe2227.pth'))))
