# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from ensemble_boxes import *
import numpy as np


def results_fusion(result,
                   result2,
                   weights=[1, 1],
                   iou_thr=0.5,
                   skip_box_thr=0.0001,
                   sigma=0.1):
    # for x1, y1, x2, y2 in boxes
    bbox_results = np.vstack(result[0])
    bboxes = bbox_results[..., :4]
    #print(bboxes)
    bboxes[..., 0] = bboxes[..., 0]/1600
    bboxes[..., 2] = bboxes[..., 2]/1600
    bboxes[..., 1] = bboxes[..., 1]/1800
    bboxes[..., 3] = bboxes[..., 3]/1800
    scores = bbox_results[..., 4]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result[0])
    ]
    labels = np.concatenate(labels)
    #print(scores)
    #print(labels)
    bbox_results2 = np.vstack(result2[0])
    bboxes2 = bbox_results2[..., :4]
    #print(bboxes)
    bboxes2[..., 0] = bboxes2[..., 0]/1600
    bboxes2[..., 2] = bboxes2[..., 2]/1600
    bboxes2[..., 1] = bboxes2[..., 1]/1800
    bboxes2[..., 3] = bboxes2[..., 3]/1800
    scores2 = bbox_results2[..., 4]
    labels2 = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result2[0])
    ]
    labels2 = np.concatenate(labels2)

    bboxes_list = [bboxes, bboxes2]
    scores_list = [scores, scores2]
    labels_list = [labels, labels2]
    
    # Weighted boxes fusion
    boxes, scores, labels = weighted_boxes_fusion(bboxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes[..., 0] = boxes[..., 0]*1600
    boxes[..., 2] = boxes[..., 2]*1600
    boxes[..., 1] = boxes[..., 1]*1800
    boxes[..., 3] = boxes[..., 3]*1800
    #print(boxes)
    #print(scores)
    #print(labels)

    fused_result = []
    for label in range(80):
        if label in labels:
            box = [np.append(boxes[i], scores[i]).tolist() for i in np.where(labels==label)[0]]
            fused_result.append(np.asarray(box, dtype=np.float32))
        else:
            fused_result.append(np.empty((0, 5), dtype=np.float32))
    #print(result)
    #print(fused_result)
    return fused_result


def multi_model_single_gpu_test(model,
                                model2,
                                data_loader,
                                data_loader2,
                                show=False,
                                out_dir=None,
                                show_score_thr=0.3):
    model.eval()
    model2.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, combined_data in enumerate(zip(data_loader, data_loader2)):
        (data, data2) = combined_data
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            result2 = model(return_loss=False, rescale=True, **data2)
        
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        if show or out_dir:
            if batch_size == 1 and isinstance(data2['img'][0], torch.Tensor):
                img_tensor = data2['img'][0]
            else:
                img_tensor = data2['img'][0].data[0]
            img_metas = data2['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model2.module.show_result(
                    img_show,
                    result2[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result2[0], tuple):
            result2 = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result2]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result2[0], dict) and 'ins_results' in result2[0]:
            for j in range(len(result2)):
                bbox_results, mask_results = result2[j]['ins_results']
                result2[j]['ins_results'] = (bbox_results,
                                             encode_mask_results(mask_results))

        fused_result = results_fusion(result, result2)
        results.extend([fused_result])

        for _ in range(batch_size):
            prog_bar.update()
    return results

