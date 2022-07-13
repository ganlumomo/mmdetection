export CUDA_VISIBLE_DEVICES=1
python tools/test.py configs/yolo/yolov3_d53_mstrain-608_273e_flir.py work_dirs/yolov3_d53_mstrain-608_273e_flir/epoch_81.pth --show-dir results/coco_trained_flir_thermal/ --eval bbox proposal --options "classwise=True"
