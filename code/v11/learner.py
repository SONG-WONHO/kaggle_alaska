import os
import gc
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import get_model
from metrics import alaska_weighted_auc


# average meter
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# loss function
def loss_func(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target)


class Learner(object):
    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.logger = None
        self.name = "model"

        if config.use_apex:
            from apex import amp, optimizers
            global amp

    def train(self, trn_data, val_data, model, optimizer, scheduler, evaluator=None):

        # dataloader
        train_loader = DataLoader(
            trn_data,
            batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.workers, pin_memory=True,
        )

        valid_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=self.config.workers, pin_memory=True,
        )

        # loger
        logger = self._create_logger()

        # training
        best_metric = 1e-8
        for epoch in range(self.config.num_epochs):
            tr_loss, tr_loss_bin = self._train_one_epoch(train_loader, model, optimizer, scheduler)
            vl_loss, vl_loss_bin, vl_metric = self._valid_one_epoch(valid_loader, model)

            # logging
            logger.loc[epoch] = [tr_loss, tr_loss_bin, vl_loss, vl_loss_bin, vl_metric, optimizer.param_groups[0]['lr']]
            logger.to_csv(os.path.join(self.config.log_path, 'log.csv'))

            # save model
            if best_metric < logger.loc[epoch, 'val_metric']:
                print(f"... From {best_metric:.4f} To {logger.loc[epoch, 'val_metric']:.4f}")
                best_metric = logger.loc[epoch, 'val_metric']
                self.best_model = copy.deepcopy(model)
                self.name = f"model.epoch_{epoch}"
                self.save()
                self.name = f"model.best"
                self.save()

            scheduler.step(metrics=vl_loss)

        self.logger = logger

    def predict(self, tst_data):
        model = self.best_model

        test_loader = DataLoader(
            tst_data,
            batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=False
        )

        pred_final = []

        model.eval()

        test_loader = tqdm(test_loader, leave=False)

        for X_batch, _, _, _ in test_loader:
            X_batch = X_batch.to(self.config.device)

            with torch.no_grad():
                preds = model(X_batch)

            preds = preds.cpu().detach()

            pred_final.append(preds)

        pred_final = torch.cat(pred_final, dim=0).numpy()

        return pred_final

    def save(self):
        if self.best_model is None:
            print("Must Train before save !")
            return

        # print(f"model will be saved here: {self.config.model_path}")

        torch.save({
            "logger": self.logger,
            "model_state_dict": self.best_model.cpu().state_dict(),
        }, f"{os.path.join(self.config.model_path, self.name)}.pt")

    def load(self, path, name=None):
        ckpt = torch.load(path)
        self.logger = ckpt['logger']
        model_state_dict = ckpt[name]
        model = get_model(self.config)
        try:
            model.load_state_dict(model_state_dict)
            print("Single GPU (Train)")
        except:
            def strip_module_str(v):
                if v.startswith('module.'):
                    return v[len('module.'):]

            model_state_dict = {strip_module_str(k): v for k, v in model_state_dict.items()}
            model.load_state_dict(model_state_dict, strict=False)
            print("Multi GPU (Train)")

        self.best_model = model.to(self.config.device)
        print("Model Loaded!")

    def _train_one_epoch(self, train_loader, model, optimizer, scheduler):
        losses = AverageMeter()

        model.train()

        train_iterator = tqdm(train_loader, leave=False)
        for X_batch, typ, y_batch, y_bin in train_iterator:
            X_batch = X_batch.to(self.config.device)
            y_bin = y_bin.to(self.config.device).type(torch.float32)

            batch_size = X_batch.size(0)

            preds = model(X_batch)

            loss = loss_func(preds.view(-1), y_bin.view(-1))
            losses.update(loss.item(), batch_size)

            optimizer.zero_grad()

            if self.config.use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            train_iterator.set_description(
                f"train bce:{losses.avg:.4f}, lr:{optimizer.param_groups[0]['lr']:.6f}")

        return 0, losses.avg

    def _valid_one_epoch(self, valid_loader, model):
        losses = AverageMeter()
        true_final, pred_final = [], []

        model.eval()

        valid_loader = tqdm(valid_loader, leave=False)
        for i, (X_batch, typ, y_batch, y_bin) in enumerate(valid_loader):
            X_batch = X_batch.to(self.config.device)
            y_bin = y_bin.to(self.config.device).type(torch.float32)

            batch_size = X_batch.size(0)

            with torch.no_grad():
                preds = model(X_batch)
                loss = loss_func(preds.view(-1), y_bin.view(-1))
                losses.update(loss.item(), batch_size)

            true_final.append(y_bin.view(-1).cpu())
            pred_final.append(preds.view(-1).detach().cpu())

            losses.update(loss.item(), batch_size)

            valid_loader.set_description(f"valid bce:{losses.avg:.4f}")

        true_final = torch.cat(true_final, dim=0).numpy()
        true_final = (true_final != 0) * 1

        pred_final = torch.cat(pred_final, dim=0)
        pred_final = nn.Sigmoid()(pred_final).numpy()

        vl_score = self._cal_metrics(pred_final, true_final)

        return 0, losses.avg, vl_score

    def _create_logger(self):
        log_cols = ['tr_loss', 'tr_loss_bin', 'val_loss', 'val_loss_bin', 'val_metric', 'lr']
        return pd.DataFrame(index=range(self.config.num_epochs), columns=log_cols)

    def _cal_metrics(self, pred, true):
        return alaska_weighted_auc(true, pred)
