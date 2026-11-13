"""Run training with checkpoint selection based on validation performance."""
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from losses import HybridObjective
from metrics import evaluate_probabilities
from models import HAM
from runtime import initialize_random_state, pair_batches, partition_pairs


@dataclass(frozen=True)
class TrainingPlan:
    data_dir: str
    output_dir: str
    epochs: int = 50
    batch_size: int = 5000
    vector_size: int = 32
    attention_size: int = 64
    heads: int = 1
    classifier_dropout: float = 0.3
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    gradient_limit: float = 1.0
    restart_period: int = 10
    restart_multiplier: int = 1
    minimum_rate: float = 1e-5
    seed: int = 1


class TrainingSession:
    def __init__(self, data, plan, accelerator):
        self.plan = plan
        self.accelerator = accelerator
        initialize_random_state(plan.seed)
        self.partition = partition_pairs(data.labelled_pairs, plan.seed)
        training_size = len(self.partition['train'])
        if min(plan.batch_size, training_size) < 2 or training_size % plan.batch_size == 1:
            raise ValueError('BatchNorm requires at least two samples in every training batch; adjust --batch-size.')
        self.batches = {
            subset: pair_batches(data.labelled_pairs[indices], plan.batch_size, training=subset == 'train')
            for subset, indices in self.partition.items()
        }
        self.network = HAM(data, plan.vector_size, plan.attention_size,
                           plan.heads, plan.classifier_dropout).to(accelerator)
        self.objective = HybridObjective(data.labelled_pairs[:, 2]).to(accelerator)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=plan.learning_rate,
                                           weight_decay=plan.weight_decay)
        self.schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=plan.restart_period, T_mult=plan.restart_multiplier,
            eta_min=plan.minimum_rate)
        tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.directory = Path(plan.output_dir) / f'ham_seed{plan.seed}_{tag}'
        self.directory.mkdir(parents=True, exist_ok=False)
        self._write_json('config.json', asdict(plan))
        np.savez(self.directory / 'split_indices.npz', **self.partition)
        covered = np.isin(data.labelled_pairs[:, 1], data.disease_graph.vertex_ids.numpy())
        self._write_json('dataset_summary.json', {
            'entity_count': len(data.catalog.categories),
            'labelled_pairs': len(data.labelled_pairs),
            'subset_sizes': {key: len(value) for key, value in self.partition.items()},
            'gene_membership_shape': list(data.gene_graph.membership.shape),
            'disease_membership_shape': list(data.disease_graph.membership.shape),
            'pairs_without_disease_neighborhood': int((~covered).sum()),
            'uncovered_disease_readout': 'Its own entity embedding; no substitution by another disease.',
            'graph_scope': 'Full graph before label-pair partition.',
            'classification_weight_scope': 'All labelled pairs.',
        })

    def _write_json(self, filename, payload):
        (self.directory / filename).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def _train_epoch(self):
        self.network.train()
        accumulated, observations = 0.0, 0
        for genes, diseases, categories in self.batches['train']:
            genes, diseases, categories = (item.to(self.accelerator) for item in (genes, diseases, categories))
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.network(genes, diseases)
            objective = self.objective(prediction, categories)
            if not torch.isfinite(objective):
                raise FloatingPointError('The training objective is not finite.')
            objective.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.plan.gradient_limit)
            self.optimizer.step()
            accumulated += objective.item() * len(categories)
            observations += len(categories)
        return accumulated / observations

    @torch.no_grad()
    def _evaluate(self, subset):
        self.network.eval()
        targets, probabilities = [], []
        accumulated, observations = 0.0, 0
        for genes, diseases, categories in self.batches[subset]:
            genes, diseases, categories = (item.to(self.accelerator) for item in (genes, diseases, categories))
            prediction = self.network(genes, diseases)
            loss = self.objective.classification(prediction.logits, categories)
            if not torch.isfinite(loss):
                raise FloatingPointError('The evaluation loss is not finite.')
            targets.append(categories.cpu().numpy())
            probabilities.append(prediction.logits.softmax(dim=1).cpu().numpy())
            accumulated += loss.item() * len(categories)
            observations += len(categories)
        return accumulated / observations, evaluate_probabilities(np.concatenate(targets), np.concatenate(probabilities))

    def run(self):
        best_validation, selected_epoch = -1.0, 0
        checkpoint = self.directory / 'best_model.pt'
        history = []
        print(f'HAM training on {self.accelerator}; outputs: {self.directory}', flush=True)
        for epoch in range(1, self.plan.epochs + 1):
            training_loss = self._train_epoch()
            self.schedule.step(epoch)
            validation_loss, validation = self._evaluate('valid')
            if validation['macro_f1'] > best_validation:
                selected_epoch, best_validation = epoch, validation['macro_f1']
                torch.save({name: value.detach().cpu() for name, value in self.network.state_dict().items()}, checkpoint)
            history.append({'epoch': epoch, 'training_loss': training_loss,
                            'validation_ce': validation_loss, 'validation_macro_f1': validation['macro_f1'],
                            'validation_therapeutic_f1': validation['therapeutic_f1'],
                            'learning_rate': self.optimizer.param_groups[0]['lr']})
            with (self.directory / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(f'Epoch {epoch}/{self.plan.epochs}: loss={training_loss:.5f}, '
                  f'validation Macro-F1={validation["macro_f1"]:.4f}', flush=True)
        self.network.load_state_dict(torch.load(checkpoint, map_location=self.accelerator, weights_only=True))
        test_loss, test_scores = self._evaluate('test')
        results = {'best_epoch': selected_epoch, 'best_valid_macro_f1': best_validation,
                   'test_ce_loss': test_loss, 'test_metrics': test_scores}
        self._write_json('metrics.json', results)
        print(json.dumps(results, indent=2), flush=True)
        return results
