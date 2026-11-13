"""Run the standard HAM training workflow."""
import argparse
from pathlib import Path

from graph_data import read_biological_data
from runtime import resolve_accelerator
from training import TrainingPlan, TrainingSession


def main():
    location = Path(__file__).resolve().parent
    command = argparse.ArgumentParser(description='HAM gene–disease association training', allow_abbrev=False)
    command.add_argument('--data-dir', default=str(location / 'datasets'))
    command.add_argument('--output-dir', default=str(location / 'runs'))
    command.add_argument('--epochs', type=int, default=50)
    command.add_argument('--batch-size', type=int, default=5000)
    command.add_argument('--vector-size', type=int, default=32)
    command.add_argument('--attention-size', type=int, default=64)
    command.add_argument('--heads', type=int, default=1)
    command.add_argument('--classifier-dropout', type=float, default=0.3)
    command.add_argument('--learning-rate', type=float, default=0.01)
    command.add_argument('--weight-decay', type=float, default=0.0)
    command.add_argument('--gradient-limit', type=float, default=1.0)
    command.add_argument('--restart-period', type=int, default=10)
    command.add_argument('--restart-multiplier', type=int, default=1)
    command.add_argument('--minimum-rate', type=float, default=1e-5)
    command.add_argument('--seed', type=int, default=1)
    command.add_argument('--device', type=int, default=0, help='-1: CPU; nonnegative: CUDA index, with CPU fallback if CUDA unavailable.')
    options = vars(command.parse_args())
    accelerator = resolve_accelerator(options.pop('device'))
    plan = TrainingPlan(**options)
    for name in ('epochs', 'batch_size', 'vector_size', 'attention_size', 'heads', 'restart_period', 'restart_multiplier'):
        if getattr(plan, name) < 1:
            command.error(f'{name} must be positive.')
    if plan.batch_size < 2 or plan.learning_rate <= 0 or plan.weight_decay < 0 or plan.gradient_limit <= 0:
        command.error('batch-size >= 2, learning-rate > 0, weight-decay >= 0 and gradient-limit > 0 are required.')
    if not 0 <= plan.classifier_dropout < 1 or not 0 <= plan.minimum_rate <= plan.learning_rate:
        command.error('classifier-dropout must be in [0,1), minimum-rate must be in [0,learning-rate].')
    data = read_biological_data(plan.data_dir)
    TrainingSession(data, plan, accelerator).run()


if __name__ == '__main__':
    main()
