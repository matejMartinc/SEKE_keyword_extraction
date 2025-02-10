# Code for experiments conducted in the paper 'SEKE: Specialised Experts for Keyword Extraction' #

## Installation, documentation ##

Published results were produced in Python 3.11 programming environment on Linux Mint 20 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager and availability of CUDA.<br/>


Clone the project from the repository.<br/>

Install dependencies if needed: pip install -r requirements.txt

We downloaded all datasets except openKP from this repository: https://gitlab.com/matej.martinc/tnt_kid

The openKP dataset is available on Hugging Face, use the "parse_openKP.py" to obtain and preprocess the data. 


### To reproduce the results published in the paper, run the code in the command line using following commands: ###

Preprocess (KP20k, Inspec, Krapivin, KPTimes or JPTimes) dataset (see 'data/inspec/inspec_test.json' folder for details on the format):<br/>

If needed, download NLTK sentence tokenizer:

```
python
import nltk
nltk.download('punkt_tab')
exit()
```
Run preprocessing script for each file you wish to preprocess:
```
python preprocessing.py --input_path 'data/inspec/inspec_test.json' --output_path 'data/inspec/inspec_TokenClassificationDataModuleFormat_test.json'
python preprocessing.py --input_path 'data/inspec/inspec_valid.json' --output_path 'data/inspec/inspec_TokenClassificationDataModuleFormat_valid.json'
```

Preprocess openKP dataset:<br/>

```
python parse_openkp.py --output_folder 'data/openKP'
```

Train and test the keyword tagger on the dataset:<br/>
```
python train.py --run_name inspec_moe_rnn --valid_path 'data/inspec/inspec_TokenClassificationDataModuleFormat_valid.json' --test_path 'data/inspec/inspec_TokenClassificationDataModuleFormat_test.json' --moe --rnn
```

If you want to use a pretrained model, additionally define the '--pretrained_path' argument, pointing to the pretrained model's checkpoint, e.g.:<br/>

```
python train.py --run_name inspec_moe_rnn --valid_path 'data/inspec/inspec_TokenClassificationDataModuleFormat_valid.json' --test_path 'data/inspec/inspec_TokenClassificationDataModuleFormat_test.json' --moe --rnn --pretrained_path /home/matej/PycharmProjects/SEKE_keyword_extraction/checkpoints/kp20k-moe-rnn-pretrained-seed-42/lightning_logs/version_0/checkpoints/epoch=3-Val_f1=0.38.ckpt
```

You can additionally define '--train_path' argument. If '--train_path' argument is not defined, we split the dataset so that 80% of documents are used for training and 20% for validation.


Use a trained keyword tagger on a dataset, e.g.:<br/>
```
python predict.py --model_path '/home/matej/PycharmProjects/SEKE_keyword_extraction/checkpoints/inspec_moe_rnn_seed_42/lightning_logs/version_0/checkpoints/epoch=3-Val_f1_10=0.55.ckpt' --test_path 'data/inspec/inspec_TokenClassificationDataModuleFormat_test.json' --output_path 'results/inspec_moe_rnn_seed_123.tsv' --moe --rnn
```

<!---
## Contributors to the code ##

Matej Martinc<br/>
Boshko Koloski<br/>
Hanh Thi Hong Tran<br/>


* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana
--->



