import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F


# Замените пути до директорий и файлов! Можете использовать для локальной отладки. 
# При проверке на сервере пути будут изменены
glue_qqp_dir = '/data/QQP/'
glove_path = '/data/glove.6B.50d.txt'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        # допишите ваш код здесь 
        return x.add(-self.mu).pow(2.0).div(2.0 * self.sigma * self.sigma).neg().exp()


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        # допишите ваш код здесь 
        for i in range(self.kernel_num - 1):
            cur_mu = (2 * (i + 1) - self.kernel_num) / (self.kernel_num - 1)
            kernels.append(GaussianKernel(
                mu=cur_mu,
                sigma=self.sigma
            ))

        kernels.append(GaussianKernel(sigma=self.exact_sigma))

        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        # допишите ваш код здесь
        if not self.out_layers:
            return torch.nn.Sequential(torch.nn.Linear(self.kernel_num, 1))

        layers = [torch.nn.Linear(self.kernel_num, self.kernel_num), torch.nn.ReLU()]
        k_in = self.kernel_num

        for k_out in self.out_layers:
            layers.append(torch.nn.Linear(k_in, k_out))
            layers.append(torch.nn.ReLU())
            k_in = k_out

        layers.append(torch.nn.Linear(k_in, 1))

        return torch.nn.Sequential(*layers)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        q = self.embeddings(query)
        d = self.embeddings(doc).transpose(1, 2)

        result = torch\
            .matmul(q, d)\
            .div(torch.linalg.norm(q, dim=2).unsqueeze(2))\
            .div(torch.linalg.norm(d, dim=1).unsqueeze(1))

        result[result.isnan()] = 0.0

        return result

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left, Embedding], [Batch, Right, Embedding]
        query, doc = inputs['query'], inputs['document']
        
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        # допишите ваш код здесь 
        return [self.vocab.get(t, self.oov_val) for t in tokenized_text]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        # допишите ваш код здесь 
        return self._tokenized_text_to_index(self.preproc_func(self.idx_to_text_mapping[str(idx)]))

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        # допишите ваш код здесь 
        triplet = self.index_pairs_or_triplets[idx]
        query = self._convert_text_idx_to_token_idxs(int(triplet[0]))
        return [
            {
                'query': query,
                'document': self._convert_text_idx_to_token_idxs(int(triplet[1]))
            },
            {
                'query': query,
                'document': self._convert_text_idx_to_token_idxs(int(triplet[2]))
            },
            triplet[3]
        ]


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        # допишите ваш код здесь 
        pair = self.index_pairs_or_triplets[idx]
        return [
            {
                'query': self._convert_text_idx_to_token_idxs(int(pair[0])),
                'document': self._convert_text_idx_to_token_idxs(int(pair[1]))
            },
            pair[2]
        ]


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels


class Solution:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)
        
        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg, 
              self.idx_to_text_mapping_dev, 
              vocab=self.vocab, oov_val=self.vocab['OOV'], 
              preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0, 
            collate_fn=collate_fn, shuffle=False)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def hadle_punctuation(self, inp_str: str) -> str:
        # допишите ваш код здесь 
        for ch in string.punctuation:
            if ch in inp_str:
                inp_str = inp_str.replace(ch, ' ')

        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        # допишите ваш код здесь

        no_punct = self.hadle_punctuation(inp_str.lower())

        tokens = nltk.word_tokenize(no_punct)

        return tokens
    
    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        # допишите ваш код здесь
        return {k: v for k, v in vocab.items() if v >= min_occurancies}
    
    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        # допишите ваш код здесь

        texts = []
        for df in list_of_df:
            texts.append(df['text_left'].unique())
            texts.append(df['text_right'].unique())

        questions = np.unique(np.concatenate(texts))

        tokens = self.simple_preproc('\n'.join(questions))

        result = list(self._filter_rare_words(Counter(tokens), min_occurancies).keys())
        return result

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        # допишите ваш код здесь
        with open(file_path, 'r') as f:
            result = {line[0]: line[1:] for line in map(lambda x: x.split(' '), f.read().splitlines())}
            return result

    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        # допишите ваш код здесь
        d = 50
        unk_words = ['PAD', 'OOV']
        vocab = {'PAD': 0, 'OOV': 1}
        matrix = [[0.0] * d, [1.0] * d]
        embs = self._read_glove_embeddings(file_path)
        np.random.seed(random_seed)

        for i, token in enumerate(inner_keys, 2):
            emb = embs.get(token)

            if emb is None:
                unk_words.append(token)
                emb = np.random.uniform(0, rand_uni_bound, d)

            matrix.append(emb)
            vocab[token] = i
        return np.array(matrix).astype(float), vocab, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        self.emb_matrix = emb_matrix
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int
                                   ) -> List[List[Union[str, float]]]:
        # допишите ваш код здесь 
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(inf_df_group_sizes[inf_df_group_sizes >= 2].index)
        groups = inp_df_select[inp_df_select.id_left.isin(glue_dev_leftids_to_use)].groupby('id_left')

        out_pairs = []
        all_docs = inp_df_select['id_right'].values

        for id_left, group in groups:
            if np.random.rand() > 0.005:
                continue

            docs = group.id_right.values
            labels = group.label.values
            ones = docs[labels == 1.0]
            zeros = docs[labels == 0.0]

            for a in ones:
                for b in zeros:
                    out_pairs.append([id_left, a, b, 1.0])

            pad = 15 - len(group)

            if pad < 1:
                continue

            for a in docs:
                for b in np.random.choice(all_docs, pad):
                    out_pairs.append([id_left, a, b, 1.0])

        # print(len(out_pairs))
        return out_pairs

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        # допишите ваш код здесь  (обратите внимание, что используются вектора numpy)
        ideal = dcg(ys_true, ys_true, 'exp2', ndcg_top_k)
        cur_dcg = dcg(ys_true, ys_pred, 'exp2', ndcg_top_k)
        return cur_dcg / ideal if ideal else 0

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])
        
        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds
        
        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()
        # допишите ваш код здесь
        self.model.sigma = 0.001
        self.model.kernels = self.model._get_kernels_layers()
        for i in range(n_epochs):
            if i % 4 == 0:
                triplets = self.sample_data_for_train_iter(self.glue_train_df, 42)
                train_dataset = TrainTripletsDataset(triplets,
                                                     self.idx_to_text_mapping_train,
                                                     vocab=self.vocab, oov_val=self.vocab['OOV'],
                                                     preproc_func=self.simple_preproc)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.dataloader_bs, num_workers=0,
                    collate_fn=collate_fn, shuffle=False)

            self.model.train()
            for batch in train_dataloader:
                inp_1, inp_2, y = batch
                opt.zero_grad()
                batch_out = self.model.forward(inp_1, inp_2)
                batch_loss = criterion(batch_out, y)
                batch_loss.backward(retain_graph=True)
                opt.step()
            # print(self.valid(self.model, self.val_dataloader))


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value

    if gain_scheme == 'exp2':
        return pow(2.0, y_value) - 1.0

    raise ValueError('incorrect gain_scheme')


def dcg(ys_true, ys_pred, gain_scheme: str, ndcg_top_k: int) -> float:
    args = np.argsort(ys_pred)[-1:-(ndcg_top_k+1):-1]
    s_true = np.take(ys_true, args)
    return sum(map(lambda p: compute_gain(p[1], gain_scheme) / np.log2(p[0]), enumerate(s_true, 2)))
