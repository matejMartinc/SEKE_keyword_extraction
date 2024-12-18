import gc
import argparse
import torch
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import seed_everything

from kw_model import TransformerModule
from transformers import AutoTokenizer
from config import TestConfig
from preprocessing import TokenClassificationDataModule


def predict(config, model_path, test_path, results_path, moe, rnn):

    seed_everything(config['seed'], workers=True)

    config['moe'] = moe
    config['rnn'] = rnn
    config['results_path'] = results_path

    print('Loading model from', model_path)
    model = TransformerModule.load_from_checkpoint(checkpoint_path=model_path, config=config)


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['pretrained_model'],
                                              add_prefix_space=True, use_fast=True)

    datamodule = TokenClassificationDataModule(
        max_length=config['max_length'],
        batch_size=config['batch_size'],
        preprocessing_num_workers=4,
        train_path=None,
        valid_path=None,
        test_path=test_path,
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        default_root_dir=model_path.split('lightning_logs')[0],
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=True),
        deterministic=True
    )

    # Evaluate the last and the best models on the test sample.
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=model_path,
    )
    del model
    del datamodule
    del trainer
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='Path to train dataset')
    parser.add_argument('--test_path', type=str, help='Path to test dataset')
    parser.add_argument('--output_path', type=str, help='Path to results .tsv file')
    parser.add_argument('--moe', action=argparse.BooleanOptionalAction)
    parser.add_argument('--rnn', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    test_config = TestConfig
    predict(test_config, args.model_path, args.test_path, args.output_path, args.moe, args.rnn)


