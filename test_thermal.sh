export CUDA_VISIBLE_DEVICES=0
python tools/test.py configs/yolo/yolov3_d53_mstrain-608_273e_flir_t.py work_dirs/flir_aligned_t_10/epoch_9.pth --show-dir results/flir_aligned_thermal/ --eval bbox proposal --options "classwise=True"
