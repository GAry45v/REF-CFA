import torch
from torchmetrics import ROC, AUROC, F1Score, AveragePrecision
import os
from torchvision.transforms import transforms
from skimage import measure
import pandas as pd
from statistics import mean
import numpy as np
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve


class Metric:
    def __init__(self,labels_list, predictions, anomaly_map_list, gt_list, config) -> None:
        self.labels_list = labels_list
        self.predictions = predictions
        self.anomaly_map_list = anomaly_map_list
        self.gt_list = gt_list
        self.config = config
        self.threshold = 0.5
    
    def image_auroc(self):
        auroc_image = roc_auc_score(self.labels_list, self.predictions)
        return auroc_image
    
    def pixel_auroc(self):
        resutls_embeddings = self.anomaly_map_list[0]
        for feature in self.anomaly_map_list[1:]:
            resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
        resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 

        gt_embeddings = self.gt_list[0]
        for feature in self.gt_list[1:]:
            gt_embeddings = torch.cat((gt_embeddings, feature), 0)

        resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
        gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

        auroc_p = AUROC(task="binary")
        
        gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
        resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()
        auroc_pixel = auroc_p(resutls_embeddings, gt_embeddings)
        return auroc_pixel
        
    def image_auprc(self):
        preds_tensor = torch.tensor(self.predictions)
        labels_tensor = torch.tensor(self.labels_list)
        auprc_metric = AveragePrecision(task="binary")
        return auprc_metric(preds_tensor, labels_tensor).item()

    def pixel_auprc(self):
        resutls_embeddings = torch.cat(self.anomaly_map_list, dim=0)
        gt_embeddings = torch.cat(self.gt_list, dim=0)

        resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min()))
        
        preds_flat = resutls_embeddings.flatten()
        gt_flat = gt_embeddings.flatten().type(torch.long)
        
        auprc_metric = AveragePrecision(task="binary")
        return auprc_metric(preds_flat.cpu(), gt_flat.cpu()).item()

    # ===================== 主要修改处 =====================
    # 移除了参数 k，改为在函数内部动态计算
    def recall_at_k(self):
        # 结合预测分数和真实标签
        combined = list(zip(self.predictions, self.labels_list))
        # 按分数从高到低排序
        combined.sort(key=lambda x: x[0], reverse=True)
        
        # 总共有多少个异常样本
        total_anomalies = sum(self.labels_list)
        if total_anomalies == 0:
            return 0.0 # 避免除以零

        # 将 K 值动态设定为异常样本的总数
        k = total_anomalies
        
        # 检查前 K 个
        top_k = combined[:k]
        
        # 计算前 K 个中有多少是真正的异常
        anomalies_in_top_k = sum(label for score, label in top_k)
        
        return anomalies_in_top_k / total_anomalies
    # =======================================================

    def optimal_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)
        youden_j = tpr - fpr
        optimal_threshold_index = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_threshold_index]
        self.threshold = optimal_threshold
        return optimal_threshold
    
    def pixel_pro(self):
        def _compute_pro(masks, amaps, num_th = 200):
            resutls_embeddings = amaps[0]
            for feature in amaps[1:]:
                resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
            amaps =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
            amaps = amaps.squeeze(1)
            amaps = amaps.cpu().detach().numpy()
            gt_embeddings = masks[0]
            for feature in masks[1:]:
                gt_embeddings = torch.cat((gt_embeddings, feature), 0)
            masks = gt_embeddings.squeeze(1).cpu().detach().numpy()
            min_th = amaps.min()
            max_th = amaps.max()
            delta = (max_th - min_th) / num_th
            binary_amaps = np.zeros_like(amaps)
            df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

            for th in np.arange(min_th, max_th, delta):
                binary_amaps[amaps <= th] = 0
                binary_amaps[amaps > th] = 1

                pros = []
                for binary_amap, mask in zip(binary_amaps, masks):
                    for region in measure.regionprops(measure.label(mask)):
                        axes0_ids = region.coords[:, 0]
                        axes1_ids = region.coords[:, 1]
                        tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                        pros.append(tp_pixels / region.area)

                inverse_masks = 1 - masks
                fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
                fpr = fp_pixels / inverse_masks.sum()

                df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)

            df = df[df["fpr"] < 0.3]
            df["fpr"] = df["fpr"] / df["fpr"].max()

            pro_auc = auc(df["fpr"], df["pro"])
            return pro_auc
        
        pro = _compute_pro(self.gt_list, self.anomaly_map_list, num_th = 200)
        return pro
    
    def miscalssified(self):
        predictions = torch.tensor(self.predictions)
        labels_list = torch.tensor(self.labels_list)
        predictions0_1 = (predictions > self.threshold).int()
        for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
            print('Sample : ', i, ' predicted as: ',p.item() ,' label is: ',l.item(),'\n' ) if l != p else None