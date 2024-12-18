"""
Takes a pretrained model with classification head and uses the peft package to do Adapter + LoRA
fine tuning.
"""
from typing import Any

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from lightning import LightningModule
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW, Optimizer
from nltk.stem.porter import PorterStemmer
from eval import eval
from typing import Optional, Tuple, Union
from torch.autograd import Variable

from transformers import AutoModelForTokenClassification, DebertaV2Model
from transformers.modeling_outputs import TokenClassifierOutput


# First define the top k router module
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # Noise logits
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


#Expert module
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
            nn.Linear(n_embd, num_labels),
        )

    def forward(self, x):
        return self.net(x)


class DebertaForSequenceLabeling(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config['num_labels']
        self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.dropout = nn.Dropout(self.config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.config['hidden_size'], self.num_labels)
        self.num_layers = 2
        if self.config['rnn']:
            print('Initializing RNN')
            self.lstm = nn.LSTM(self.config['hidden_size'], self.config['hidden_size'], dropout=0.3, num_layers=self.num_layers,
                                bidirectional=True)
            self.lstm_dropout = nn.Dropout(0.3)

        if self.config['moe']:
            print('Initializing MOE')
            self.router = NoisyTopkRouter(self.config['hidden_size'], self.config['num_experts'], self.config['top_k'])
            self.experts = nn.ModuleList([Expert(self.config['hidden_size'], self.config['hidden_size']) for _ in range(self.config['num_experts'])])
            self.top_k = self.config['top_k']

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.config['hidden_size']).to(next(self.parameters()).device))
        c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.config['hidden_size']).to(next(self.parameters()).device))
        return (h0, c0)


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:

        return_dict = self.config['use_return_dict']

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        #print(sequence_output.shape)

        sequence_output = self.dropout(sequence_output)
        if self.config['moe']:
            gating_output, indices = self.router(sequence_output)
            logits = torch.zeros([sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2]], dtype=sequence_output.dtype, layout=sequence_output.layout, device=sequence_output.device)

            # Reshape inputs for batch processing
            flat_x = sequence_output.view(-1, sequence_output.size(-1))
            flat_gating_output = gating_output.view(-1, gating_output.size(-1))

            # Process each expert in parallel
            for i, expert in enumerate(self.experts):
                # Create a mask for the inputs where the current expert is in top-k
                expert_mask = (indices == i).any(dim=-1)
                flat_mask = expert_mask.view(-1)

                if flat_mask.any():
                    expert_input = flat_x[flat_mask]
                    expert_output = expert(expert_input)

                    # Extract and apply gating scores
                    gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                    weighted_output = expert_output * gating_scores

                    # Update final output additively by indexing and adding
                    logits[expert_mask] += weighted_output.squeeze(1)
            sequence_output = logits
        if self.config['rnn']:
            lstm_hidden = self.init_hidden(sequence_output.size(0))
            embeddings = self.lstm_dropout(sequence_output).permute(1, 0, 2)
            lstm_out, self.lstm_hidden = self.lstm(embeddings, lstm_hidden)
            lstm_out = (lstm_out[:, :, :self.config['hidden_size']] + lstm_out[:, :, self.config['hidden_size']:])
            lstm_out = lstm_out.permute(1, 0, 2)
            sequence_output = sequence_output + lstm_out
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class TransformerModule(LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config

        model = DebertaForSequenceLabeling(config=config)
        peft_config = LoraConfig(
            target_modules=['value_proj', 'query_proj'],
            #target_modules=['value', 'query'], # use for for roberta
            modules_to_save=["classifier", "score", "lstm", 'router', 'experts.0', 'experts.1', 'experts.2', 'experts.3'],
            #task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='all'
        )


        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()

        self.lr = config['lr']
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.stemmer = PorterStemmer()
        config_dict = {}
        for att in dir(self.config):
            if not att.startswith('__'):
                config_dict[att] = getattr(self.config, att)
        self.save_hyperparameters(config_dict)


    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
    ):

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        predictions = torch.argmax(outputs["logits"], dim=2)
        predictions = predictions.detach().cpu().numpy().tolist()
        for i in range(len(batch['labels'])):
            labels = batch['labels'][i]
            kw_all = batch['meta'][i]['kw_all']
            kw_in_paper = batch['meta'][i]['kw_in_paper']
            text = batch['meta'][i]['text']
            word_ids = batch['meta'][i]['word_ids']
            self.validation_step_outputs.append((predictions[i], labels, kw_all, kw_in_paper, text, word_ids))
        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return predictions, batch

    def on_validation_epoch_end(self):
        all_preds, all_true = [], []
        for predictions, labels, kw_all, kw_in_paper, text, word_ids in self.validation_step_outputs:
            preds, true = self._extract_keywords(predictions, labels, kw_all, kw_in_paper, text, word_ids)
            all_preds.append(preds)
            all_true.append(true)
        p_1, r_1, f_1, p_3, r_3, f_3, p_5, r_5, f_5, p_10, r_10, f_10, p_k, r_k, f_k, p_M, r_M, f_M = eval(all_preds, all_true)
        self.log("Val_f1_10", f_10, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Val_p_10", p_10, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Val_r_10", r_10, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Val_f1_5", f_5, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Val_p_5", p_5, on_step=False, on_epoch=True, sync_dist=True)
        self.log("Val_r_5", r_5, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        predictions = torch.argmax(outputs["logits"], dim=2)
        predictions = predictions.detach().cpu().numpy().tolist()
        for i in range(len(batch['labels'])):
            labels = batch['labels'][i]
            kw_all = batch['meta'][i]['kw_all']
            kw_in_paper = batch['meta'][i]['kw_in_paper']
            text = batch['meta'][i]['text']
            word_ids = batch['meta'][i]['word_ids']
            self.test_step_outputs.append((predictions[i], labels, kw_all, kw_in_paper, text, word_ids))
        return predictions, batch

    def on_test_epoch_end(self):
        results = []
        all_preds, all_true = [], []
        for predictions, labels, kw_all, kw_in_paper, text, word_ids in self.test_step_outputs:
            preds, true = self._extract_keywords(predictions, labels, kw_all, kw_in_paper, text, word_ids)
            all_preds.append(preds)
            all_true.append(true)
            results.append((";".join(true), ";".join(preds), " ".join(text)))
        if self.trainer.is_global_zero:
            p_1, r_1, f_1, p_3, r_3, f_3, p_5, r_5, f_5, p_10, r_10, f_10, p_k, r_k, f_k, p_M, r_M, f_M = eval(all_preds, all_true)
            self.log("Test_f1_10", f_10, on_step=False, on_epoch=True)
            self.log("Test_p_10", p_10, on_step=False, on_epoch=True)
            self.log("Test_r_10", r_10, on_step=False, on_epoch=True)
            self.log("Test_f1_5", f_5, on_step=False, on_epoch=True)
            self.log("Test_p_5", p_5, on_step=False, on_epoch=True)
            self.log("Test_r_5", r_5, on_step=False, on_epoch=True)
            self.log("Test_f1_1", f_1, on_step=False, on_epoch=True)
            self.log("Test_p_1", p_1, on_step=False, on_epoch=True)
            self.log("Test_r_1", r_1, on_step=False, on_epoch=True)
            self.log("Test_f_3", f_3, on_step=False, on_epoch=True)
            self.log("Test_p_3", p_3, on_step=False, on_epoch=True)
            self.log("Test_r_3", r_3, on_step=False, on_epoch=True)
            df = pd.DataFrame(results, columns=['true', 'preds', 'text'])
            df.to_csv(self.config['results_path'], index=False, sep='\t')


    def _extract_keywords(self, predictions, labels, kw_all, kw_in_paper, text, word_ids):
        keywords = []
        keyword = []
        for idx, pred in enumerate(predictions):
            if pred==1:
                if keyword:
                    keywords.append(" ".join(keyword))
                    keyword = []
                if idx==0 or word_ids[idx-1] != word_ids[idx]:
                    if word_ids[idx] is not None:
                        keyword.append(text[word_ids[idx]])
            elif pred==2:
                if idx != 0 and word_ids[idx - 1] != word_ids[idx]:
                    if word_ids[idx] is not None:
                        keyword.append(text[word_ids[idx]])
            else:
                if keyword and word_ids[idx - 1] != word_ids[idx]:
                    keywords.append(" ".join(keyword))
                    keyword = []
        if keyword:
            keywords.append(" ".join(keyword))

        stemmed_keywords = set()
        filtered_keywords = []
        punctuation = "!#$%&'()*+,.:;<=>?@[\]^_`{|}~"
        for kw in keywords:
            has_punct = False
            for punct in punctuation:
                if punct in kw:
                    has_punct = True
                    break
            if not has_punct:
                stemmed_kw = " ".join([self.stemmer.stem(word) for word in kw.split()])
                if stemmed_kw not in stemmed_keywords:
                    filtered_keywords.append(kw)
                    stemmed_keywords.add(stemmed_kw)
        return list(set(filtered_keywords))[:10], kw_in_paper

    def configure_optimizers(self):
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=0.0,
        )