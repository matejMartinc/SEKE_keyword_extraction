import gc
import os
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import seed_everything
from kw_model import TransformerModule
from transformers import AutoTokenizer
from config import TrainConfig
from preprocessing import TokenClassificationDataModule
import argparse


def train(run_config, run_name, train_path, valid_path, test_path, moe, rnn, pretrained_path):

    run_config['moe'] = moe
    run_config['rnn'] = rnn

    dc = {
         'train_path': train_path,
         'valid_path': valid_path,
         'test_path': test_path,
         'batch_size': run_config['batch_size'],
         'model_checkpoint_dir': os.path.join('./checkpoints', run_name),
         'results_path': os.path.join('./results', run_name)
    }

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    if not os.path.exists('./results'):
        os.makedirs('./results')

    seed = run_config['seed']
    seed_everything(seed, workers=True)
    print("Training on", dc['valid_path'], 'seed:', seed)
    run_config['batch_size'] = dc['batch_size']
    run_config['model_checkpoint_dir'] = dc['model_checkpoint_dir'] + '_seed_' + str(seed)
    run_config['results_path'] = dc['results_path'] + '_seed_' + str(seed) + '.tsv'


    if len(pretrained_path) > 0:
        print('Loading model from', pretrained_path)
        model = TransformerModule.load_from_checkpoint(checkpoint_path=pretrained_path, config=run_config)
    else:
        print('Not using pretrained model')
        model = TransformerModule(
            config=run_config
        )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=run_config['pretrained_model'],
                                              add_prefix_space=True, use_fast=True)

    datamodule = TokenClassificationDataModule(
        max_length=run_config['max_length'],
        batch_size=run_config['batch_size'],
        preprocessing_num_workers=4,
        train_path=dc['train_path'],
        valid_path=dc['valid_path'],
        test_path=dc['test_path'],
        tokenizer=tokenizer,
    )

    # Keep the model with the highest F1 score.
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{Val_f1_10:.2f}",
        monitor="Val_f1_10",
        mode="max",
        verbose=True,
        save_top_k=1,
    )

    # Run the training loop.
    trainer = Trainer(
        callbacks=[
            EarlyStopping(
                monitor="Val_f1_10",
                min_delta=run_config['min_delta'],
                patience=run_config['patience'],
                verbose=True,
                mode="max",
            ),
            checkpoint_callback,
        ],
        default_root_dir=run_config['model_checkpoint_dir'],
        fast_dev_run=bool(run_config['debug_mode_sample']),
        max_epochs=run_config['max_epochs'],
        max_time=run_config['max_time'],
        #precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=True),
        deterministic=True
    )
    trainer.fit(model=model, datamodule=datamodule)
    best_model_path = checkpoint_callback.best_model_path

    trainer = Trainer(
        default_root_dir=run_config['model_checkpoint_dir'],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=True),
        deterministic=True
    )

    # Evaluate the last and the best models on the test sample.
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=best_model_path,
    )
    del model
    del datamodule
    del trainer
    gc.collect()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='inspec_moe_rnn', help='Path to train dataset')
    parser.add_argument('--train_path', type=str, default='', help='Path to train dataset')
    parser.add_argument('--valid_path', type=str, help='Path to valid dataset')
    parser.add_argument('--test_path', type=str, help='Path to test dataset')
    parser.add_argument('--pretrained_path', default='', type=str, help='Path to pretrained model checkpoint')
    parser.add_argument('--moe', action=argparse.BooleanOptionalAction)
    parser.add_argument('--rnn', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    torch.cuda.empty_cache()
    gc.collect()

    # Train model
    train(TrainConfig, args.run_name, args.train_path, args.valid_path, args.test_path, args.moe, args.rnn, args.pretrained_path)


