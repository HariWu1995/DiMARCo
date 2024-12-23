# import sys
# sys.path.append('./src')

import os
from pathlib import Path

import src.arckit as arckit
import src.utils as utils
from src.trainer.dimarco import ModelTrainer


## Configuration
augmented = True
categorical = True
padding_value = -1

data_name = 'iarc' + ('_cat' if categorical else ('_aug' if augmented else ''))
model_name = 'unet'
trainer_name = 'dimarco'
training_folder = f'./results/{trainer_name}-{data_name}-{model_name}'

if not os.path.isdir(training_folder):
    Path(training_folder).mkdir(parents=True, exist_ok=True)

dataset_config = dict(
       num_classes = 10,
        batch_size = -1, 
         grid_size = [32, 32],
    normalize_size = False, 
      padding_mode = 'center',
     padding_value = padding_value,
         augmented = augmented,
       categorical = categorical,
)

model_config = dict(
    backbone = model_name, 
    num_stages = 3, init_filters = 64,
    num_classes = 10, background_class = padding_value,
    layered_input = categorical,
)

trainer_config = dict(
    loss_fn = 'huber',  # 2D: huber, 3D: mse-layered
    optimizer = 'adam', 
    lr_schedule = 'cosine', lr = 0.000369, 
    num_epochs = 10, 
    accum_steps = 4,
    grad_max_norm = [6.9, 1.69], 
)

callback_config = dict(save_folder = training_folder)
diffuser_config = dict(noise_schedule = 'beta', denoising_steps = 20, train_noise = 0.888)
distributed_config = dict(precision_type = 'fp16', distributed_batches = True)

all_config = dict(  data = dataset_config, 
                    model = model_config, 
                    trainer = trainer_config, 
                    diffuser = diffuser_config, )

utils.save_yaml(os.path.join(training_folder, 'model-config.yaml'), all_config)


## Load data
dataset_dir = Path('./data/competition')

train_challenges = utils.load_json(dataset_dir / 'arc-agi_training_challenges.json')
train_solutions  = utils.load_json(dataset_dir / 'arc-agi_training_solutions.json')

eval_challenges = utils.load_json(dataset_dir / 'arc-agi_evaluation_challenges.json')
eval_solutions  = utils.load_json(dataset_dir / 'arc-agi_evaluation_solutions.json')

# test_challenges = utils.load_json(dataset_dir / 'arc-agi_test_challenges.json')


## Format data: combine train & eval datasets
task_set = arckit.format_data(train_challenges, train_solutions, 
                               eval_challenges,  eval_solutions)

task_set = arckit.load_data(task_set, eval=False, combine=True)

if dataset_config.pop('categorical', False):
    from src.datasets import iARCDatasetDepth as Dataset

elif dataset_config.pop('augmented', False):
    from src.datasets import iARCDatasetAug as Dataset

else:
    from src.datasets import iARCDatasetNaive as Dataset

dataset = Dataset(task_set=task_set, **dataset_config)
print('\n\n Dataset size:', len(dataset))


## Training

trainer = ModelTrainer( train_dataloader = dataset, 
                        **model_config, **diffuser_config, 
                        **trainer_config, **distributed_config, 
                        **callback_config, )
trainer.train()
