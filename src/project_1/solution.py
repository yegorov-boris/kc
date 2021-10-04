import string
import json
import math
from typing import Dict, List
import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
from langdetect import detect, LangDetectException
import nltk
import numpy as np
import torch
import faiss


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return x.add(-self.mu).pow(2.0).div(2.0 * self.sigma * self.sigma).neg().exp()


class KNRM(torch.nn.Module):
    def __init__(self, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()

        embs = torch.load(os.environ.get('EMB_PATH_KNRM'))
        self.embeddings = torch.nn.Embedding(embs['weight'].shape[0], 50)
        self.embeddings.load_state_dict(embs)
        self.embeddings.eval()

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        mlp = torch.load(os.environ.get('MLP_PATH'))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(self.kernel_num, 1))
        self.mlp.load_state_dict(mlp)
        self.mlp.eval()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num - 1):
            cur_mu = (2 * (i + 1) - self.kernel_num) / (self.kernel_num - 1)
            kernels.append(GaussianKernel(
                mu=cur_mu,
                sigma=self.sigma
            ))

        kernels.append(GaussianKernel(sigma=self.exact_sigma))

        return kernels

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        q = self.embeddings(query)
        d = self.embeddings(doc).transpose(1, 2)

        result = torch \
            .matmul(q, d) \
            .div(torch.linalg.norm(q, dim=2).unsqueeze(2)) \
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


def handle_punctuation(inp_str: str) -> str:
    for ch in string.punctuation:
        if ch in inp_str:
            inp_str = inp_str.replace(ch, ' ')

    return inp_str


def simple_preproc(inp_str: str) -> List[str]:
    return nltk.word_tokenize(handle_punctuation(inp_str.lower()))


executor = ThreadPoolExecutor(1)
app = Flask(__name__)
state = {
    'init_finished': False,
    'index_finished': False,
    'mapping': []
}
raw_embs = dict()


def make_embs():
    with open(os.environ.get('EMB_PATH_GLOVE'), 'r') as f:
        for line in map(lambda x: x.split(' '), f.read().splitlines()):
            raw_embs[line[0]] = list(map(float, line[1:]))

        f.close()

    state['knrm'] = KNRM(sigma=0.001)

    with open(os.environ.get('VOCAB_PATH'), 'r') as f:
        state['vocab'] = json.loads(f.read())
        f.close()

    state['init_finished'] = True
    print('----------- init')


# make_embs()


def to_vector(q, pad=False):
    tokens = simple_preproc(q)
    vector = [v for vs in map(raw_embs.get, tokens) if vs is not None for v in vs]

    return [state['vocab'].get(t, 1) for t in tokens], pad_vector(vector) if pad else vector


def pad_vector(vector):
    return vector[:state['max_dim']] + [0.0] * (state['max_dim'] - len(vector))


@app.route('/ping')
def ping():
    return {'status': 'ok'} if state['init_finished'] else ''


@app.route('/update_index', methods=['POST'])
def update_index():
    state['documents'] = request.json['documents']
    vectors = []
    i = 0

    for k, q in state['documents'].items():
        # try:
        #     if detect(q) != 'en':
        #         continue
        # except LangDetectException:
        #     continue

        t, v = to_vector(q)
        vectors.append(v)
        state['mapping'].append({
            'id': k,
            'text': q,
            'tokens': t,
        })
        i += 1

    state['max_dim'] = max(map(len, vectors))
    padded_vectors = np.array([pad_vector(v) for v in vectors]).astype('float32')
    x = int(2 * math.sqrt(i))
    index_description = f'IVF{x},Flat'
    state['index'] = faiss.index_factory(state['max_dim'], index_description)
    state['index'].train(padded_vectors)
    state['index'].add(padded_vectors)
    state['index_finished'] = True

    return {
        'status': 'ok',
        'index_size': state['index'].ntotal
    }


@app.route('/query', methods=['POST'])
def handle_query():
    if not state['index_finished']:
        return {'status': 'FAISS is not initialized!'}

    result = {
        'lang_check': [],
    }
    padded = []
    tokens = []

    for q in request.json['queries']:
        is_en = False

        try:
            is_en = detect(q) == 'en'
        except LangDetectException:
            pass

        result['lang_check'].append(is_en)

        if not is_en:
            continue

        t, v = to_vector(q, True)
        padded.append(v)
        tokens.append(t)

    _, groups = state['index'].search(np.array(padded).astype('float32'), 10)
    sgs = []

    for g, t in zip(groups, tokens):
        n = len(g)
        ts = [state['mapping'][id]['tokens'] for id in g]
        suggestions = np.array([(state['mapping'][id]['id'], state['mapping'][id]['text']) for id in g])
        m = max(map(len, ts))
        preds = state['knrm'].predict({
            'query': torch.Tensor(t).repeat(n, 1).int(),
            'document': torch.Tensor([tid + [0] * (m - len(tid)) for tid in ts]).int()
        })
        sgs.append(suggestions[np.argsort(preds.detach().numpy())[::-1]])

    result['suggestions'] = []

    for is_en in result['lang_check']:
        result['suggestions'].append(sgs.pop(0).tolist() if is_en else None)

    return result


executor.submit(make_embs)
