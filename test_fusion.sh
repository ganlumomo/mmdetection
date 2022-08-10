export CUDA_VISIBLE_DEVICES=0
python tools/test_fusion.py configs/yolo/yolov3_d53_mstrain-608_273e_flir_rgbt.py work_dirs/yolov3_d53_mstrain-608_273e_flir_t/epoch_81.pth --checkpoint2 work_dirs/yolov3_d53_mstrain-608_273e_flir_rgb/epoch_79.pth --show-dir results/flir_aligned_fusion/ --eval bbox proposal --options "classwise=True"
