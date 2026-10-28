"""Reproducible setup helpers for formal-v1. No training on import."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
VERSION = 'mosi-word-downstream-v3-replicates-dev-20261020'

@contextmanager
def isolated_rng():
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)

def preserve_rng(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with isolated_rng():
            return fn(*args, **kwargs)
    return wrapped

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def tensor_digest(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def backbone_digest(model):
    return tensor_digest((k, v) for k, v in model.state_dict().items()
                         if not k.startswith('corr_model.') and not k.endswith('.w_rel')
                         and k != 'constant_scales')

def seeded_loader(dataset, batch_size, shuffle, seed):
    from torch.utils.data import DataLoader
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)

def strict_cache(pkl_path, dataset, split, bert_path, cache_root, max_seq_length,
                 batch_size=16, device=None):
    """Read frozen caches only. Formal runs never replace shared inputs."""
    path = Path(cache_root) / dataset / (split + '.fixed_contextual_word_v1.pt')
    cached = torch.load(path, map_location='cpu')
    meta = cached['metadata']
    expected = {'pipeline_version': 'fixed_contextual_word_v1', 'dataset': dataset,
                'split': split, 'max_seq_length': max_seq_length,
                'pkl_sha256': file_sha256(pkl_path),
                'bert_config_sha256': file_sha256(Path(bert_path) / 'config.json'),
                'text_layer': 'last_hidden_state', 'pooling': 'mean_subwords',
                'special_tokens': '[CLS],[SEP] excluded'}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('Frozen cache mismatch: {} {}'.format(path, key))
    return cached, path, False

def verify_inputs(root, dataset):
    manifest = json.loads((Path(root) / 'protocol/input_manifest.json').read_text())
    selected = [item for item in manifest['files'] if item['dataset'] in [dataset, 'shared']]
    for item in selected:
        if file_sha256(item['path']) != item['sha256']:
            raise RuntimeError('Frozen input changed: ' + item['path'])
    return {item['path']: item['sha256'] for item in selected}

def verify_source(root):
    manifest = json.loads((Path(root) / 'protocol/frozen_source.json').read_text())
    for name, digest in manifest['files'].items():
        if file_sha256(Path(root) / name) != digest:
            raise RuntimeError('Frozen source changed: ' + name)
    return manifest


def verify_environment(root):
    import platform
    from importlib.metadata import version
    expected = json.loads((Path(root) / 'protocol/environment.json').read_text())
    current = {'python': platform.python_version(),
               'packages': {name: version(name) for name in expected['packages']}}
    if current != expected:
        raise RuntimeError('Runtime versions differ from the frozen protocol')
    return current
