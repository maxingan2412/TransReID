import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


import os
import shutil
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    visual = True
    if visual:
        # 定义根目录
        data_root = "/home/ma1/work/data/market1501/query/"
        visorg_dir = "/home/ma1/work/TransReID/visorg"
        visattention_dir = "/home/ma1/work/TransReID/jpm1125pinghua"

        # 创建目标文件夹
        os.makedirs(visorg_dir, exist_ok=True)
        os.makedirs(visattention_dir, exist_ok=True)

        from scipy.ndimage import gaussian_filter

        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
            with torch.no_grad():
                # 将输入数据迁移到设备
                img = img.to(device)
                camids = camids.to(device)
                target_view = target_view.to(device)

                # 计算特征
                feat = model(img, cam_label=camids, view_label=target_view)

                for i in range(img.shape[0]):
                    image_path = os.path.join(data_root, imgpath[i])
                    feature_tensor = feat[i].cpu()

                    # 在 768 维度上取平均，得到 (210,)
                    #feature_avg = feature_tensor.mean(dim=1)
                    # 计算权重
                    # weights = torch.softmax(feature_tensor, dim=1)
                    #
                    # # 加权平均
                    # feature_avg = (feature_tensor * weights).sum(dim=1)
                    # 使用自适应平均池化
                    feature_avg = F.adaptive_avg_pool1d(feature_tensor.unsqueeze(1), output_size=1).squeeze(1)

                    # 对输入特征进行归一化处理
                    feature_avg = (feature_avg - feature_avg.min()) / (feature_avg.max() - feature_avg.min())

                    # 根据特征的形状选择重塑方式
                    if feature_avg.shape[0] == 210:
                        # 直接重塑为 (21, 10)
                        feature_map = feature_avg.view(21, 10)
                    elif feature_avg.shape[0] == 128:
                        # 重塑为 (16, 8)
                        feature_map = feature_avg.view(16, 8)

                        # 插值到 (21, 10)
                        feature_map = torch.nn.functional.interpolate(
                            feature_map.unsqueeze(0).unsqueeze(0), size=(21, 10), mode='bilinear', align_corners=False
                        )[0, 0]
                    else:
                        raise ValueError(f"Unexpected feature_avg shape: {feature_avg.shape[0]}")

                    # 插值到最终形状 (256, 128)
                    attention_map_upsampled = torch.nn.functional.interpolate(
                        feature_map.unsqueeze(0).unsqueeze(0), size=(256, 128), mode='bilinear', align_corners=False
                    )[0, 0]

                    attention_map_upsampled = F.avg_pool2d(attention_map_upsampled.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0, 0]

                    # 归一化注意力图
                    attention_map_upsampled = (attention_map_upsampled - attention_map_upsampled.min()) / (
                            attention_map_upsampled.max() - attention_map_upsampled.min()
                    )

                    # **叠加到原图**
                    original_image = cv2.imread(image_path)
                    original_image = cv2.resize(original_image, (128, 256))
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                    # 转换注意力图为伪彩色
                    attention_map_numpy = attention_map_upsampled.cpu().numpy()  # 转换为 NumPy 数组
                    heatmap = cv2.applyColorMap((attention_map_numpy * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                    # 混合原图和热力图
                    combined = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

                    # **保存注意力图**
                    attention_path = os.path.join(visattention_dir,
                                                  os.path.splitext(os.path.basename(image_path))[0] + ".png")
                    plt.imsave(attention_path, combined)

        print("所有图片和注意力图处理完成！")

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)






    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


