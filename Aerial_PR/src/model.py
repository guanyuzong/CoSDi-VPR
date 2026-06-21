# ============================================================
# ExpA: backbone LR=0.25×, freeze backbone at epoch 8
# ============================================================

import torch
import lightning as L
from pytorch_metric_learning import losses, miners
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src import utils


class BoQModel(L.LightningModule):
    def __init__(
            self,
            backbone,
            aggregator,
            lr=1e-4,
            lr_mul=0.1,
            weight_decay=1e-3,
            warmup_epochs=10,
            milestones=[10, 20],
            silent=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.lr = lr
        self.lr_mul = lr_mul
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.silent = silent

        # ========== ExpA: backbone LR=0.25×, freeze at epoch 8 ==========
        self.backbone_lr_mul = 0.25
        self.freeze_backbone_epoch = 10
        self._backbone_frozen = False

        self.ms_loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.)
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=0.1)
        self._did_val_vis = False
        self._vis_slots = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        self._vis_layer = 0
        self.mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    def configure_optimizers(self):
        no_decay_keywords = ['queries', 'log_tau', 'bias']
        agg_decay = [p for n, p in self.aggregator.named_parameters()
                     if p.requires_grad and not any(nd in n for nd in no_decay_keywords)]
        agg_no_decay = [p for n, p in self.aggregator.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay_keywords)]

        optimizer_params = [
            {"params": self.backbone.parameters(), "lr": self.lr * self.backbone_lr_mul, "weight_decay": self.weight_decay},
            {"params": agg_decay,                  "lr": self.lr,       "weight_decay": self.weight_decay},
            {"params": agg_no_decay,               "lr": self.lr,       "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mul
        )
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        if not self._backbone_frozen and self.current_epoch >= self.freeze_backbone_epoch:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self._backbone_frozen = True
            # backbone param group的LR设为0
            optimizer = self.optimizers()
            optimizer.param_groups[0]['lr'] = 0.0
            print(f"\n[Epoch {self.current_epoch}] Backbone FROZEN\n")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = (self.trainer.global_step + 1) / total_warmup_steps
            lr_scale = min(1.0, lr_scale)
            for pg in optimizer.param_groups:
                if self._backbone_frozen and pg is optimizer.param_groups[0]:
                    pg['lr'] = 0.0
                    continue
                initial_lr = pg.get("initial_lr", pg["lr"])
                pg["lr"] = lr_scale * initial_lr
        elif self._backbone_frozen:
            optimizer.param_groups[0]['lr'] = 0.0
        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)

    def off_diag_mean_sq(self, G):
        B, Q, _ = G.shape
        mask = ~torch.eye(Q, device=G.device, dtype=torch.bool)
        off = G[:, mask]
        return (off ** 2).mean()

    def loss_rep_on_slots(self, slots_all):
        s = F.normalize(slots_all, dim=-1)
        G = s @ s.transpose(1, 2)
        return self.off_diag_mean_sq(G)

    def compute_loss(self, descriptors, labels, attentions, slots_all, Hf=None, Wf=None):
        mined_pairs = self.ms_miner(descriptors, labels)
        loss_main = self.ms_loss(descriptors, labels, mined_pairs)
        loss_rep = self.loss_rep_on_slots(slots_all)
        A_fg2 = F.normalize(attentions, p=2, dim=-1)
        G = A_fg2 @ A_fg2.transpose(1, 2)
        Qf = attentions.size(1)
        eye = torch.eye(Qf, device=attentions.device, dtype=torch.bool).unsqueeze(0)
        G_off = G.masked_fill(eye, 0)
        loss_rep_atten = torch.relu(G_off - 0.3).mean()
        loss = loss_main + 0.3 * loss_rep_atten + loss_rep
        self.log("loss_main", loss_main, prog_bar=True, logger=True)
        self.log("loss_rep", loss_rep, prog_bar=True, logger=True)
        self.log("loss_rep_atten", 0.3 * loss_rep_atten, prog_bar=True, logger=True)
        return loss

    def forward(self, x, return_feats=False):
        x = self.backbone(x)
        descriptors, slots_all, attentions, (Hf, Wf) = self.aggregator(x, return_feats)
        return descriptors, slots_all, attentions, (Hf, Wf)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.flatten(0, 1)
        labels = labels.flatten()
        descriptors, slots_all, attentions, (Hf, Wf) = self(images)
        loss = self.compute_loss(descriptors, labels, attentions, slots_all, Hf=Hf, Wf=Wf)
        return loss

    def on_train_epoch_end(self):
        self.trainer.train_dataloader.dataset._refresh_dataframes()

    def on_validation_epoch_start(self):
        self.validation_outputs = {}
        self._did_val_vis = False
        val_loader = self.trainer.val_dataloaders[0]
        num_batches = len(val_loader)
        self._vis_batch_idx = torch.randint(0, num_batches, (1,)).item()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, _ = batch
        descriptors, slots_all, attentions, (Hf, Wf) = self(images)
        descriptors = descriptors.detach().cpu()
        if dataloader_idx not in self.validation_outputs:
            self.validation_outputs[dataloader_idx] = []
        self.validation_outputs[dataloader_idx].append(descriptors)

    def on_validation_epoch_end(self):
        val_dataloaders = self.trainer.val_dataloaders
        recalls = {}
        for dataloader_idx, descriptors_list in self.validation_outputs.items():
            descriptors = torch.cat(descriptors_list, dim=0)
            dataset = val_dataloaders[dataloader_idx].dataset
            if self.trainer.fast_dev_run:
                if dataloader_idx == 0:
                    print("\nFast dev run: skipping recall@k computation\n")
            else:
                recalls_dict = utils.compute_recall_performance(
                    descriptors, dataset.num_references, dataset.num_queries,
                    dataset.ground_truth, k_values=[1, 5, 10, 15],
                )
                recalls_log = {
                    f"{dataset.dataset_name}/R@1": recalls_dict[1],
                    f"{dataset.dataset_name}/R@5": recalls_dict[5],
                }
                recalls[dataset.dataset_name] = recalls_dict
                self.log_dict(recalls_log, prog_bar=False, logger=True)
        if recalls and not self.silent:
            utils.display_recall_performance(list(recalls.values()), list(recalls.keys()))
        self.validation_outputs.clear()
