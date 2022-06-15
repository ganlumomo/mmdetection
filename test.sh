export CUDA_VISIBLE_DEVICES=1
python tools/test.py configs/yolo/yolov3_d53_mstrain-608_273e_flir.py checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth --show-dir results/coco_pretrained_flir_rgb/ --eval bbox proposal --options "classwise=True"
