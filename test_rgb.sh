export CUDA_VISIBLE_DEVICES=0
python tools/test.py configs/yolo/yolov3_d53_mstrain-608_273e_flir_rgb.py work_dirs/flir_aligned_rgb_10/epoch_10.pth --show-dir results/flir_aligned_rgb/ --eval bbox proposal --options "classwise=True"
