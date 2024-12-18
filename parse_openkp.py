from datasets import load_dataset
import json
import os
from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='Path to trained byte pair encoding model')
    args = parser.parse_args()

    # get entire dataset
    dataset = load_dataset("midas/openkp", "raw")

    # sample from the train split
    print("Sample from training data split")
    train_sample = dataset["train"][0]
    print("Fields in the sample: ", [key for key in train_sample.keys()])
    print("Tokenized Document: ", train_sample["document"])
    print("Document BIO Tags: ", train_sample["doc_bio_tags"])
    print("Extractive/present Keyphrases: ", train_sample["extractive_keyphrases"])
    print("Abstractive/absent Keyphrases: ", train_sample["abstractive_keyphrases"])
    print("\n-----------\n")

    splits = ['train', 'validation', 'test']
    iob2num = {'B': 1, 'I': 2, 'O': 0}
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    path = os.path.join(args.output_folder, 'openKP_TokenClassificationDataModuleFormat_')
    for split in splits:
        output_path = path + split + '.json'
        out = open(output_path, 'w', encoding='utf')
        num_files = 0
        num_tokens = 0
        num_kw = 0
        for sample in dataset[split]:
            tokens = sample["document"]
            tags = sample["doc_bio_tags"]
            tags = [iob2num[x] for x in tags]
            kw_in_paper = sample["extractive_keyphrases"]
            keywords = kw_in_paper + train_sample["abstractive_keyphrases"]
            json_doc = json.dumps({"label_tags": tags, "tokens": tokens, "kw_all": list(set(keywords)), "kw_in_paper": list(set(kw_in_paper))})
            out.write(json_doc + '\n')
            num_files += 1
            num_tokens += len(tokens)
            num_kw += len(kw_in_paper)
        out.close()
        print(f'Done processing, num files: {num_files}, avg num tokens: {num_tokens / num_files}, avg num kw: {num_kw / num_files}')


