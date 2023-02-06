import argparse
import re

import pandas as pd
import torch.utils.data
from omegaconf import OmegaConf
from tokenizers.implementations import BertWordPieceTokenizer, SentencePieceBPETokenizer, SentencePieceUnigramTokenizer

from projection import Projection
import json
import numpy as np
from datasets import load_dataset
import pytorch_lightning as pl
from omegaconf.dictconfig import DictConfig
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from typing import Any, Dict, List


class NLPDataModule(pl.LightningDataModule):

    def __init__(self, modelcfg: DictConfig, datasetcfg: DictConfig, dataset: str, **kwargs):
        super(NLPDataModule, self).__init__(**kwargs)
        self.modelcfg = modelcfg
        self.datasetcfg = datasetcfg
        print(datasetcfg.dataset_type)
        self.vocab_cfg = datasetcfg.vocab
        self.projecion_cfg = modelcfg.projection
        self.projecion = Projection(self.vocab_cfg.vocab_path, self.projecion_cfg.feature_size, self.projecion_cfg.window_size)
        self.tokenizer = BertWordPieceTokenizer(**self.vocab_cfg.tokenizer)

    def get_dataset_cls(self):
        if self.datasetcfg.dataset_type.name == 'mtop':
            return MtopDataset
        if self.datasetcfg.dataset_type.name == 'matis':
            return MultiAtisDataset
        if self.datasetcfg.dataset_type.name == 'imdb':
            return ImdbDataset
        if self.datasetcfg.dataset_type.name == 'sst2':
            return SST2Dataset
        if self.datasetcfg.dataset_type.name == 'sst5':
            return SST5Dataset
        if self.datasetcfg.dataset_type.name == 'agnews':
            return AGDataset
        if self.datasetcfg.dataset_type.name == 'snli':
            return SNLIDataset
        if self.datasetcfg.dataset_type.name == 'qnli':
            return QNLIDataset
        if self.datasetcfg.dataset_type.name == 'yelp2':
            return YelpDataset
        if self.datasetcfg.dataset_type.name == 'yelp5':
            return Yelp5Dataset
        if self.datasetcfg.dataset_type.name == 'qqp':
            return QQPDataset
        if self.datasetcfg.dataset_type.name == 'rte':
            return RTEDataset
        if self.datasetcfg.dataset_type.name == 'cola':
            return CoLADataset
        if self.datasetcfg.dataset_type.name == "hyperpartisan":
            return HyperpartisanDataset
        if self.datasetcfg.dataset_type.name == "amazon":
            return AmazonDataset
        if self.datasetcfg.dataset_type.name == "dbpedia":
            return dbpediaDataset
        if self.datasetcfg.dataset_type.name == "semeval_task_1":
            return semeval1

    def setup(self, stage: str = None):
        root = Path(self.datasetcfg.dataset_type.dataset_path)
        label_list = Path(self.datasetcfg.dataset_type.labels).read_text().splitlines() \
            if isinstance(self.datasetcfg.dataset_type.labels, str) \
            else self.datasetcfg.dataset_type.labels
        self.label_map = {label: index for index, label in enumerate(label_list)}
        dataset_cls = self.get_dataset_cls()
        print(dataset_cls)
        if stage in (None, 'fit'):
            self.train_set = dataset_cls(root, 'train', self.datasetcfg.dataset_type.max_seq_len, self.tokenizer,
                                         self.projecion, self.label_map)
            if self.datasetcfg.dataset_type.name in ['imdb', "yelp2", 'yelp5', 'agnews', 'dbpedia', "amazon"]:
                mode = 'test'
            else:
                mode = 'validation'
            self.eval_set = dataset_cls(root, mode, self.datasetcfg.dataset_type.max_seq_len, self.tokenizer, self.projecion,
                                        self.label_map)
        if stage in (None, 'test'):
            if self.datasetcfg.dataset_type.name in ['imdb', "yelp2", 'agnews', 'dbpedia', "amazon", "semeval_task_1",
                                                     'yelp5']:
                mode = 'test'
            else:
                mode = 'validation'
            self.test_set = dataset_cls(root, mode, self.datasetcfg.dataset_type.max_seq_len, self.tokenizer, self.projecion,
                                        self.label_map)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.modelcfg.loader.batch_size, shuffle=True,
                          num_workers=self.modelcfg.loader.num_workers, persistent_workers=True)  # , pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.eval_set, self.modelcfg.loader.batch_size, shuffle=True,
                          num_workers=self.modelcfg.loader.num_workers, persistent_workers=True)  # , pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.modelcfg.loader.batch_size, shuffle=True,
                          num_workers=self.modelcfg.loader.num_workers, persistent_workers=True)  # , pin_memory=True)


class PnlpMixerDataset(Dataset):
    def __init__(self, max_seq_len: int, tokenizer: Tokenizer, projection: Projection, label_map: Dict[str, int],
                 **kwargs):
        super(PnlpMixerDataset, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.projection = projection
        self.max_seq_len = max_seq_len
        self.label_map = label_map

    def normalize(self, text: str) -> str:
        return text.replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"')

    def project_features(self, words: List[str]) -> np.ndarray:
        encoded = self.tokenizer.encode(words, is_pretokenized=True, add_special_tokens=False)
        tokens = [[] for _ in range(len(words))]
        for index, token in zip(encoded.words, encoded.tokens):
            tokens[index].append(token)
        # print(tokens)
        features = self.projection(tokens)
        # print(features)
        padded_featrues = np.pad(features, ((0, self.max_seq_len - len(words)), (0, 0)))
        return padded_featrues

    def get_words(self, fields: List[str]) -> List[str]:
        raise NotImplementedError

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, index) -> Dict[str, Any]:
        # print(index)
        fields = self.data[index]
        # print(fields)
        words = self.get_words(fields)
        # print(words)
        features = self.project_features(words)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }


class YelpDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(YelpDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('yelp_polarity', 'plain_text', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !") \
            .replace("\n", " ")

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class Yelp5Dataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(Yelp5Dataset, self).__init__(*args, **kwargs)
        self.data = load_dataset("yelp_review_full", split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !") \
            .replace("\n", " ")

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class AmazonDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(AmazonDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset("amazon_polarity", split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !") \
            .replace("\n", " ")

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["content"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class dbpediaDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(dbpediaDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset("dbpedia_14", split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !") \
            .replace("\n", " ")

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["content"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class ImdbDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(ImdbDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('imdb', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class HyperpartisanDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(HyperpartisanDataset, self).__init__(*args, **kwargs)
        data = load_dataset("hyperpartisan_news_detection", "byarticle", split="train")
        print(data)
        train_size = int(len(data) * 0.8)
        val_size = int(len(data) * 0.1)
        test_size = len(data)-train_size-val_size
        train, val, test = torch.utils.data.random_split(data, [train_size, val_size, test_size])
        if filename == "train":
            self.data = train
        if filename == "test":
            self.data = test
        if filename == 'validation':
            self.data = val
        self.label_map = {False: 0, True: 1}

    def __len__(self) -> int:
        return len(self.data)

    def len_compute(self):
        return

    def normalize(self, text: str) -> str:
        html_label = "<[^>]+>"
        email = "^[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)"
        chars = "&#[0-9]*"
        url = "(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?"
        text = text.replace('’', '\'') \
            .replace('–', '-') \
            .replace('‘', '\'') \
            .replace('´', '\'') \
            .replace('“', '"') \
            .replace('”', '"') \
            .replace('<splt>', '  ').replace('&#[0-9]*; ', "")
        text = re.sub(html_label, "", text)
        text = re.sub(url, "", text)
        text = re.sub(email, "", text)
        text = re.sub(chars, "", text)
        return text

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(self.label_map[fields["hyperpartisan"]])


class SST2Dataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(SST2Dataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'sst2', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class SST5Dataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(SST5Dataset, self).__init__(*args, **kwargs)
        self.data = load_dataset("SetFit/sst5", split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])



class CoLADataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(CoLADataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'cola', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class AGDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(AGDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('ag_news', split=filename)

    def __len__(self) -> int:
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace('<br />', ' ')

    def get_words(self, fields: List[str]) -> List[str]:
        return [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["text"]))][
               :self.max_seq_len]

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(fields["label"])


class QNLIDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(QNLIDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'qnli', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: List[str]) -> List[str]:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["question"]))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence"]))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"])

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }


class QQPDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(QQPDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'qqp', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: List[str]) -> List[str]:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["question1"]))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["question2"]))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"])

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }


class semeval1(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(semeval1, self).__init__(*args, **kwargs)
        self.data = load_dataset("sem_eval_2014_task_1", split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: List[str]) -> List[str]:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["premise"]))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["hypothesis"]))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["entailment_judgment"])

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }



class RTEDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(RTEDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('glue', 'rte', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: List[str]) -> List[str]:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence1"]))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields["sentence2"]))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"])

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }


class SNLIDataset(PnlpMixerDataset):
    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(SNLIDataset, self).__init__(*args, **kwargs)
        self.data = load_dataset('snli', split=filename)

    def __len__(self):
        return len(self.data)

    def normalize(self, text: str) -> str:
        return text.replace("\\", " ") \
            .replace("?", " ?") \
            .replace(".", " .") \
            .replace(",", " ,") \
            .replace("!", " !")

    def get_words(self, fields: List[str]) -> List[str]:
        return [[w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields['premise']))][
                :self.max_seq_len],
                [w[0] for w in self.tokenizer.pre_tokenizer.pre_tokenize_str(self.normalize(fields['hypothesis']))][
                :self.max_seq_len]]

    def compute_labels(self, fields):
        return np.array(fields["label"]+1)

    def __getitem__(self, index) -> Dict[str, Any]:
        fields = self.data[index]
        words = self.get_words(fields)
        u = self.project_features(words[0]).reshape(1, self.max_seq_len, -1)
        v = self.project_features(words[1]).reshape(1, self.max_seq_len, -1)
        features = np.concatenate((u, v), axis=0)
        labels = self.compute_labels(fields)
        return {
            'inputs': features,
            'targets': labels
        }


class MultiAtisDataset(PnlpMixerDataset):

    def __init__(self, root: Path, filename: str, *args, **kwargs) -> None:
        super(MultiAtisDataset, self).__init__(*args, **kwargs)
        self.data = []
        for file in root.glob(f'{filename}_*.tsv'):
            self.data.extend(file.read_text().splitlines()[1:-1])

    def __len__(self):
        return len(self.data)

    def get_words(self, fields: List[str]) -> List[str]:
        return self.normalize(fields[1]).split(' ')

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        return np.array(self.label_map[fields[-1]])


class MtopDataset(PnlpMixerDataset):

    def __init__(self, root: Path, filename: str, *args, **kwargs):
        super(MtopDataset, self).__init__(*args, **kwargs)
        self.data = []
        for file in root.glob(f'*/{filename}.txt'):
            self.data.extend(file.read_text().splitlines())

    def __len__(self) -> int:
        return len(self.data)

    def get_words(self, fields: List[str]) -> List[str]:
        segments = json.loads(fields[-1])
        normalized_words = [self.normalize(word) for word in segments['tokens']]
        return normalized_words

    def compute_labels(self, fields: List[str]) -> np.ndarray:
        segments = json.loads(fields[-1])
        num_words = len(segments['tokens'])
        slot_list = fields[2].split(',')
        slot = np.ones([num_words], dtype=np.long) * self.label_map['O']
        slot = np.pad(slot, (0, self.max_seq_len - num_words), constant_values=-1)
        starts = {}
        ends = {}
        for index, span in enumerate(segments['tokenSpans']):
            starts[span['start']] = index
            ends[span['start'] + span['length']] = index + 1
        for s in slot_list:
            if not s:
                break
            start, end, _, val = s.split(':', maxsplit=3)
            start_index = starts[int(start)]
            end_index = ends[int(end)]
            slot[start_index] = self.label_map[f'B-{val}']
            if end_index > start_index + 1:
                slot[start_index + 1:end_index] = self.label_map[f'I-{val}']
        return slot