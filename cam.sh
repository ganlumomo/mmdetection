# GradCAM: https://github.com/open-mmlab/mmdetection/pull/7987/
python demo/vis_cam.py configs/yolo/yolov3_d53_mstrain-608_273e_flir.py work_dirs/yolov3_d53_mstrain-608_273e_flir/epoch_81.pth --no-norm-in-bbox --img-folder /groups/ARCL/FLIR_ADAS_v2/images_thermal_val/data/ --method gradcam --target-layers bbox_head.convs_bridge[0].conv --out-dir results/cam/
