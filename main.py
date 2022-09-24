from pytorch_lightning import seed_everything

from TCADatasetModules.CV_DatasetModule import CVDataModule
from TCADatasetModules.NLP_DatasetModule import NLPDataModule
from models import TCAMixerImageCls, TCAMixerNLPCls
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Any, Dict, List, Optional
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class TCAMixerCVTrainModule(pl.LightningModule):
    def __init__(self, mode: str, optimizer_cfg: DictConfig, embedding_cfg: DictConfig, backbone_cfg: DictConfig,
                 classification_cfg: DictConfig, **kwargs):
        super(TCAMixerCVTrainModule, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = TCAMixerImageCls(
            mode,
            embedding_cfg,
            backbone_cfg,
            classification_cfg
        )

    def metrics(self, outputs):
        ACC, Macro_Precision, Macro_F1, Macro_recall, length = 0, 0, 0, 0, len(outputs)
        for output in outputs:
            ACC += accuracy_score(output['targets'], output['predicts'])
            Macro_Precision += precision_score(output['targets'], output['predicts'], average='macro')
            Macro_F1 += f1_score(output['targets'], output['predicts'], average='macro')
            Macro_recall += recall_score(output['targets'], output['predicts'], average='macro')
        return {
            "ACC": ACC / length,
            "Macro_Precision": Macro_Precision / length,
            "Macro_F1": Macro_F1 / length,
            "Macro_Recall": Macro_recall / length
        }

    def shared_step(self, batch, batch_idx):
        x, targets = batch['image'], batch["label"]
        outs = self.model(x)
        loss = F.cross_entropy(outs, targets.long())
        predict = torch.argmax(outs, dim=1)
        return {
            'batch_ids': batch_idx,
            'predicts': predict.detach().cpu().numpy(),
            'targets': targets.detach().cpu().numpy(),
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, batch_idx)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return results

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('train_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        results = self.shared_step(batch, batch_idx)
        self.log('val_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return results

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('val_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        results = self.shared_step(batch, batch_idx)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('test_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.AdamW(self.parameters(), **optimizer_cfg)
        return optimizer


class TCAMixerNLPTrainModule(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, dataset_cfg: DictConfig, **kwargs):
        super(TCAMixerNLPTrainModule, self).__init__(**kwargs)
        self.optimizer_cfg = model_cfg.optimizer
        self.model = TCAMixerNLPCls(
            "nlp",
            model_cfg,
            dataset_cfg
        )

    def metrics(self, outputs):
        ACC, Macro_Precision, Macro_F1, Macro_recall, length = 0, 0, 0, 0, len(outputs)
        for output in outputs:
            ACC += accuracy_score(output['targets'], output['predicts'])
            Macro_Precision += precision_score(output['targets'], output['predicts'], average='macro')
            Macro_F1 += f1_score(output['targets'], output['predicts'], average='macro')
            Macro_recall += recall_score(output['targets'], output['predicts'], average='macro')
        return {
            "ACC": ACC / length,
            "Macro_Precision": Macro_Precision / length,
            "Macro_F1": Macro_F1 / length,
            "Macro_Recall": Macro_recall / length
        }

    def shared_step(self, batch, batch_idx):
        x, targets = batch['inputs'], batch["targets"]
        # print(x[0])
        outs = self.model(x)
        # print(outs.shape)
        loss = F.cross_entropy(outs, targets.long())
        predict = torch.argmax(outs, dim=1)
        return {
            'batch_ids': batch_idx,
            'predicts': predict.detach().cpu().numpy(),
            'targets': targets.detach().cpu().numpy(),
            'loss': loss
        }

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, batch_idx)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return results

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('train_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        results = self.shared_step(batch, batch_idx)
        # self.log('val_loss', results['loss'], on_step=False, on_epoch=False, prog_bar=False, logger=True)
        return results

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('val_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        results = self.shared_step(batch, batch_idx)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accuracy = self.metrics(outputs)
        self.log('test_acc', accuracy['ACC'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pre', accuracy["Macro_Precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', accuracy["Macro_F1"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', accuracy['Macro_Recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-n', '--name', type=str)
    args.add_argument('-a', '--architecture', type=str)
    args.add_argument('-p', '--ckpt', type=str)
    args.add_argument('-t', '--train', type=str, default="train")
    args.add_argument('-d', "--dataset", type=str, default="mnist")
    args.add_argument('-i', '--index', type=int, default=7)
    return args.parse_args()


def get_module_cls(dataset: str):
    if dataset in ["cifar10", "cifar100", "imagenet", "mnist", 'fashion', 'flowers102']:
        return TCAMixerCVTrainModule, CVDataModule
    if dataset in ["agnews", 'amazon', 'dbpedia', 'hyperpartisan', 'imdb', 'yelp2', 'sst2', 'cola', 'qqp']:
        return TCAMixerNLPTrainModule, NLPDataModule
    else:
        raise "More Tasks support later...."


if __name__ == '__main__':
    seed_everything(1)
    args = parse_args()
    modelcfg = OmegaConf.load("configs/model.yml")
    print(modelcfg)
    loader_cfg = modelcfg.loader
    backbone_cfg = modelcfg.backbone
    optimizer_cfg =modelcfg.optimizer
    if args.dataset in ["cifar10", "cifar100", "imagenet", "mnist", 'fashion', 'flowers102']:
        datasetcfg = OmegaConf.load("configs/cv/"+args.dataset+".yml")
    if args.dataset in ["agnews", 'amazon', 'dbpedia', 'hyperpartisan', 'imdb', 'yelp2', 'sst2', 'cola', 'qqp']:
        datasetcfg = OmegaConf.load("configs/nlp/"+args.dataset+".yml")
    print(datasetcfg)
    modules = get_module_cls(args.dataset)
    data_module = modules[1](modelcfg, datasetcfg, args.dataset)
    # data_module.setup('fit')
    # train_set = data_module.train_set
    # print(train_set.__getitem__(0))

    #  NLPDataModule 调试完成

    if args.ckpt:
        train_module = modules[0].load_from_checkpoint(args.ckpt, model_cfg=modelcfg, dataset_cfg=datasetcfg)
    else:
        train_module = modules[0](model_cfg=modelcfg, dataset_cfg=datasetcfg)
    print(train_module)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_acc',
                filename='mixer-best-{epoch:03d}-{val_acc:.3f}-{val_f1:.3f}',
                save_top_k=1,
                mode='max',
                save_last=True
            ),
            # pl.callbacks.early_stopping.EarlyStopping(
            #     monitor="val_acc",
            #     min_delta=0.001,
            #     mode='max'
            # )
        ],
        enable_checkpointing=True,
        gpus=1,
        log_every_n_steps=2,
        logger=pl.loggers.TensorBoardLogger("logs/", args.dataset),
        max_epochs=100,
        check_val_every_n_epoch=1,
        # limit_train_batches=0.5,
        # limit_val_batches=0.1
    )
    if args.train == 'train':
        trainer.fit(train_module, data_module)
    if args.train == 'test':
        trainer.test(train_module, data_module, ckpt_path=args.ckpt)
