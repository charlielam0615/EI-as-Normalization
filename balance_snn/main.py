import argparse
import importlib
import datetime
import os, time, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid, json
from tqdm import trange

import brainpy as bp
import brainpy.math as bm

from model import SNN
from train import get_MNIST_data, Trainer
from utils import epoch_visual

# current time in string
_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# parse arguments
parser = argparse.ArgumentParser(description='LIF MNIST Training')
parser.add_argument('-config', type=str, default='configs/base_cpu.py', help='config file') 
parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
parser.add_argument('-exp-id', type=str, default='_'.join([_date, str(uuid.uuid4())[:8]]), help='experiment name') 
args = parser.parse_args()
print(f"using configs: {args.config}")

all_config = importlib.import_module(args.config.replace('/', '.').replace('.py', ''))
model_config = all_config.model_config
global_config = all_config.global_config

# make output dir
out_dir = os.path.join(args.out_dir, args.exp_id)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# dump config into json
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(
        {
            "model_config": model_config.__dict__, 
            "global_config": global_config.__dict__,
        }, 
        f, indent=4
    )

# download data
bm.set_platform(global_config.device)
bm.set_environment(mode=bm.training_mode, dt=1.)
x_train, y_train, x_test, y_test = get_MNIST_data('./data/MNIST')

# build model, set optimizer and loss function
model = SNN(model_config)
optimizer = bp.optim.Adam(lr=global_config.lr, train_vars=model.train_vars().unique())
trainer = Trainer(model, optimizer, global_config)

# visualization
visual_inputs = model.encoder(x_test[0: 1], num_step=global_config.T)
epoch_visual(model, visual_inputs, 0, out_dir)
# start training
max_test_acc = 0.
for epoch_i in range(global_config.epochs):
    t0 = time.time()
    train_loss, train_acc = trainer.train_epoch(x_train, y_train)

    # visualization
    epoch_visual(model, visual_inputs, epoch_i+1, out_dir)

    # validation
    test_loss, test_acc = trainer.validate_epoch(x_test, y_test)

    t = (time.time() - t0) / 60
    print(f'epoch {epoch_i}, used {t:.3f} min, '
        f'train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, '
        f'test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}')

    # save checkpoint
    if max_test_acc < test_acc:
        max_test_acc = test_acc
        states = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch_i': epoch_i,
            'train_acc': train_acc,
            'test_acc': test_acc,
        }
        bp.checkpoints.save_pytree(os.path.join(out_dir, 'mnist-lif.bp'), states)


# testing
state_dict = bp.checkpoints.load_pytree(os.path.join(out_dir, 'mnist-lif.bp'))
model.load_state_dict(state_dict['net'])
_, test_acc = trainer.validate_epoch(x_test, y_test)
print('Max test accuracy: ', test_acc)





