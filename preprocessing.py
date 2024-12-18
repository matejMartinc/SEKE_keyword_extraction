import numpy
import torch
from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
import json
from nltk.stem.porter import PorterStemmer

from torch.utils.data import DataLoader, Dataset

from lightning import LightningDataModule
from typing import Optional
import random
import argparse



class TokenClassificationDataModule(LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, max_length, batch_size, preprocessing_num_workers, tokenizer, train_path, valid_path, test_path):
        super().__init__()
        self.max_len = max_length
        self.batch_size = batch_size
        self.preprocessing_num_workers = preprocessing_num_workers
        self.tokenizer = tokenizer
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_data = []
            valid_data = []
            if len(self.train_path) > 0:
                with open(self.train_path) as f:
                    counter = 0
                    lines = f.readlines()
                    random.Random(42).shuffle(lines)
                    for line in lines:
                        counter += 1
                        #if counter > 1000:
                        #    break
                        j_content = json.loads(line)
                        train_data.append(j_content)

                with open(self.valid_path) as f:
                    for line in f:
                        counter += 1
                        # if counter > 1000:
                        #    break
                        j_content = json.loads(line)
                        valid_data.append(j_content)
            else:
                with open(self.valid_path) as f:
                    counter = 0
                    lines = f.readlines()
                    print(len(lines))
                    val_idx = int(0.8 * len(lines))
                    random.Random(42).shuffle(lines)
                    for line in lines:
                        counter += 1
                        # if counter > 1000:
                        #    break
                        j_content = json.loads(line)
                        if counter <= val_idx:
                            train_data.append(j_content)
                        else:
                            valid_data.append(j_content)

            print('Len train data:', len(train_data))
            print('Len valid data:', len(valid_data))
            self.train = TokenClassDataset(train_data, self.tokenizer, self.max_len)
            self.valid = TokenClassDataset(valid_data, self.tokenizer, self.max_len)


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_data = []
            counter = 0
            with open(self.test_path) as f:
                for line in f:
                    counter += 1
                    #if counter > 1000:
                    #    break
                    j_content = json.loads(line)
                    test_data.append(j_content)
            self.test = TokenClassDataset(test_data, self.tokenizer, self.max_len)
            print('Len test data:', len(test_data))


    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.preprocessing_num_workers, collate_fn=metadatacollator)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.preprocessing_num_workers, collate_fn=metadatacollator)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.preprocessing_num_workers, collate_fn=metadatacollator)


def metadatacollator(batch):
    return {
        'meta': [x['meta'] for x in batch],
        'labels': torch.stack([x['labels'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
    }


class TokenClassDataset(Dataset):

  def __init__(self, docs, tokenizer, max_len):
    self.docs = docs
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.docs)

  def tokenize_and_align_labels(self, examples):
      tokenized_inputs = self.tokenizer(examples["tokens"], max_length=self.max_len, padding="max_length", truncation=True, is_split_into_words=True, add_special_tokens=True, return_tensors='pt',)
      label = examples[f"label_tags"]
      word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
      previous_word_idx = None
      label_ids = []

      for word_idx in word_ids:  # Set the special tokens to -100.
          if word_idx is None:
              label_ids.append(-100)
          elif word_idx != previous_word_idx:  # Only label the first token of a given word.
              label_ids.append(label[word_idx])
          else:
              label_ids.append(-100)
          previous_word_idx = word_idx
      if len(label_ids) <= self.max_len:
          padding = [-100] * (self.max_len - len(label_ids))
          label_ids = label_ids + padding
      elif len(label_ids) > self.max_len:
          label_ids = label_ids[:self.max_len]
      label_ids = torch.LongTensor(label_ids)
      tokenized_inputs["labels"] = label_ids
      tokenized_inputs['kw_all'] = examples["kw_all"]
      tokenized_inputs['kw_in_paper'] = examples["kw_in_paper"]
      tokenized_inputs['text'] = examples["tokens"]
      tokenized_inputs['word_ids'] = word_ids
      return tokenized_inputs

  def __getitem__(self, item):
    doc = self.docs[item]
    encoding = self.tokenize_and_align_labels(doc)
    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': encoding['labels'].flatten(),
      'meta': {'kw_all': encoding['kw_all'], 'kw_in_paper': encoding['kw_in_paper'],
               'text': encoding['text'], 'word_ids': encoding['word_ids']}
    }


def file_to_seq_class_format(input_path, output_path):
    print('Processing doc: ', input_path)
    stemmer = PorterStemmer()
    counter = 0
    out = open(output_path, 'w', encoding='utf')
    num_tokens = 0
    num_files = 0
    num_kw = 0
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f:
            counter += 1
            if counter % 10000 == 0:
                print('Processing json: ', counter)
            line = json.loads(line)
            title = line.get('title') or ''
            abstract = line.get('abstract') or ''

            text = title + '. ' + abstract
            #fulltext = line.get("fulltext") or ''
            #text = text + ' ' + fulltext
            try:
                kw = line['keywords']
            except:
                kw = line['keyword']
            if isinstance(kw, list):
                kw = ";".join(kw)
            json_doc = tokenize_doc(text, kw, stemmer)
            out.write(json_doc + '\n')
            num_files += 1
            num_tokens += len(json.loads(json_doc)['tokens'])
            num_kw += len(json.loads(json_doc)['kw_in_paper'])
    out.close()
    print(f'Done processing, num files: {num_files}, avg num tokens: {num_tokens/num_files}, avg num kw: {num_kw/num_files}')


def tokenize_doc(text, keywords, stemmer):
    words = preprocess_text(text)
    kws = []
    keywords = keywords.lower()
    keywords = keywords.replace('-', ' ')
    keywords = keywords.replace('/', ' ')
    keywords = keywords.replace('∗', ' ')
    keywords = keywords.split(';')
    for kw in keywords:
        kw = kw.split()
        kws.append(kw)

    not_in_text = defaultdict(int)
    present_kw = 0
    all_kw = 0

    length = len(words)
    labels = numpy.zeros(length)

    kw_in_paper = set()
    stemmed_kw_in_paper = set()

    for j, word in enumerate(words):
        for kw in kws:
            lkw = len(kw)
            is_keyword = False
            if j + lkw < length:
                for k in range(lkw):
                    w = words[j + k]
                    if stemmer.stem(w.lower()) != stemmer.stem(kw[k].lower()):
                        break
                else:
                    is_keyword = True
            if is_keyword:
                for k in range(lkw):
                    if k == 0:
                        labels[j + k] = 1
                    else:
                        labels[j + k] = 2

                kw_in_paper.add(" ".join(kw))
                stemmed_kw = " ".join([stemmer.stem(w.lower()) for w in kw])
                stemmed_kw_in_paper.add(stemmed_kw)

    #remove keywords that don't appear
    num_all_kw = len(kws)
    not_kws = [" ".join(x) for x in kws if " ".join(x) not in kw_in_paper]
    kws = [x for x in kws if " ".join(x) in kw_in_paper]

    for k in not_kws:
        not_in_text[k] += 1
    all_kw += num_all_kw
    present_kw += len(kws)
    return json.dumps({"label_tags":labels.astype(int).tolist(), "tokens":words, "kw_all": list(set(keywords)), "kw_in_paper": list(kw_in_paper)})


def preprocess_text(text):
    words = []
    text = text.replace('-', ' ')
    text = text.replace('/', ' ')
    text = text.replace('∗', ' ')
    for sent in sent_tokenize(text):
        sent = word_tokenize(sent)
        words.extend([w.lower() for w in sent])
    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to data in json form')
    parser.add_argument('--output_path', type=str, help='Path to trained byte pair encoding model')
    args = parser.parse_args()
    file_to_seq_class_format(args.input_path, args.output_path)







