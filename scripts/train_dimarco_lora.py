# import sys
# sys.path.append('./src')

import os
from copy import deepcopy
from pathlib import Path

import pandas as pd

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import src.arckit as arckit
import src.utils as utils
from src.models import build_model
from src.trainer.dimarco_lora import LoraTrainer as Trainer


## Configuration
augmented = True
categorical = True
padding_value = -1

data_name = 'iarc' + ('_cat' if categorical else ('_aug' if augmented else ''))
model_name = 'unet'
trainer_name = 'dimarco'
training_folder = f'./results/{trainer_name}-{data_name}-{model_name}'

assert os.path.isdir(training_folder), \
    f"Cannot access {training_folder}, please check where the base-model locates ..."

all_config = utils.load_yaml(
                os.path.join(training_folder, 'model-config.yaml'))

dataset_config = all_config['data']
trainer_config = all_config['trainer']
diffuser_config = all_config['diffuser']
basemodel_config = all_config['model']

lora_folder = os.path.join(training_folder, 'lora')

if not os.path.isdir(lora_folder):
    Path(lora_folder).mkdir(parents=True, exist_ok=True)

callback_config = dict(save_folder = lora_folder)
distributed_config = dict(precision_type = 'fp16', distributed_batches = True)

model_config = dict(
    backbone = model_name,
    layered_input = categorical,

    adapter = 'lora',
    adapter_rank = 64,
    adapter_weight = 1.,
)

trainer_config.update(dict(
    num_steps = 100, 
    eval_step = 5,
    lr = 0.00169, 
    grad_max_norm = 1.69, 
))

del trainer_config['num_epochs']
del trainer_config['accum_steps']

all_config = dict(  data = dataset_config, 
                    model = model_config, 
                    trainer = trainer_config, 
                    diffuser = diffuser_config, )

utils.save_yaml(os.path.join(training_folder, 'lora-config.yaml'), all_config)


## Load data
dataset_dir = Path('./data/competition')

train_challenges = utils.load_json(dataset_dir / 'arc-agi_training_challenges.json')
train_solutions  = utils.load_json(dataset_dir / 'arc-agi_training_solutions.json')

eval_challenges = utils.load_json(dataset_dir / 'arc-agi_evaluation_challenges.json')
eval_solutions  = utils.load_json(dataset_dir / 'arc-agi_evaluation_solutions.json')

# test_challenges = utils.load_json(dataset_dir / 'arc-agi_test_challenges.json')


## Format data
task_set = arckit.format_data(train_challenges, train_solutions, 
                               eval_challenges,  eval_solutions)

train_set, \
 eval_set = arckit.load_data(task_set, eval=True, combine=False)

if dataset_config.pop('categorical', False):
    from src.datasets import ARCDatasetDepth as Dataset
else:
    from src.datasets import ARCDatasetNaive as Dataset


## Load base-model

def load_model():
    model = build_model(**basemodel_config)
    params = torch.load(
                os.path.join(training_folder, 'model-best.pt'), map_location = DEVICE)

    if 'model' in params.keys():
        params = params['model']

    model.load_state_dict(params, strict=True)

    return model

## Training

trainer_log = dict(taskId = [], subset = [], loss_train = [], loss_eval = [])

for subset, taskset in [('train', train_set), ('eval', eval_set)]:

    for tid in range(len(taskset)):

        task = taskset[tid]
        task_id = task.id
        
        print('\n'*3, '-'*19)
        print(f'{subset.capitalize()} - {tid+1:03d} / {len(taskset)} - {task_id}')
        model_config['model'] = load_model()
    
        dloader = Dataset(task_set = task, grid_size = [32, 32])
        trainer = Trainer(task_id = task_id, 
                          task_loader = dloader, **model_config,
                                                **trainer_config, **diffuser_config, 
                                                **callback_config, **distributed_config, )
        trainer.train()

        loss_train, loss_eval = trainer.get_best_result()

        trainer_log['taskId'].append(task.id)
        trainer_log['subset'].append(subset)
        trainer_log['loss_eval'].append(loss_eval)
        trainer_log['loss_train'].append(loss_train)


## Logging

results_df = pd.DataFrame.from_dict(trainer_log)
results_df.to_csv(training_folder + '/lora-losses.csv', index=False)

