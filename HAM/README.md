# HAM

A PyTorch implementation of hypergraph attention with learnable hyperedge weights and hybrid learning for gene-disease association prediction.

### Install requirements

Tested with Python 3.14. Dependency versions are listed in `requirements.txt`.

```sh
pip install -r requirements.txt
```

### Run HAM

```sh
python exp.py
```

For CPU training, use `python exp.py --device -1`. Run `python exp.py --help` for all options.

Place `datasets.pkl` and `unique_node_sets.pkl` in `datasets/`. Training results are saved to `runs/` by default.
