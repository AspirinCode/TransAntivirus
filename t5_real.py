from transformers import (
    AdamW,
    DataCollatorWithPadding,
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)
from iupac_dataset import IUPACDataset
from torch.utils.data import DataLoader
import os
import tempfile
import re
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import os.path as pt
from model_toy import Model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
MAXLEN=128

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

@dataclass
class DatasetArguments:
    dataset_dir: str = field(
            metadata={"help": "Directory where dataset is locaed"}
    )
    iupac_vocab_fn: str = field(
            metadata={"help": "File containing iupac sentencepiece model"}
    )
    smile_vocab_fn: str = field(
            metadata={"help": "File containing smile sentencepiece model"}
    )
    dataset_filename: str = field(
            default="iupacs_logp.txt",
            metadata={"help": "Name of dataset file in dataset_dir"}
    )
    mask_probability: float = field(
            default=0.15,
            metadata={"help": "Fraction of tokens to mask"}
    )
    mean_span_length: int = field(
            default=5,
            metadata={"help": "Max contiguous span of tokens to mask"}
    )
    name_col: str = field(
            default="IUPACName",
            metadata={"help": "Header of column that contains the names"}
    )
    prepend_target: bool = field(
            default=True,
            metadata={"help": "Prepend names with discretized targets?"}
    )
    target_col: str = field(
            default="aLogP",
            metadata={"help": "Header of column that contains the target vals"}
    )
    dataset_filename: str = field(
            default="iupacs_logp.txt",
            metadata={"help": "Filename containing data"}
    )
    low_cutoff: float = field(
            default=-0.4,
            metadata={"help": "Cutoff between <low> and <med>"}
    )
    high_cutoff: float = field(
            default=5.6,
            metadata={"help": "Cutoff between <med> and <high>"}
    )


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
            default=None,
            metadata={"help": "Checkpoint to start training from"}
    )
    tokenizer_type: Optional[str] = field(
            default="IUPAC",
            metadata={"help": "How to tokenize chemicals (SMILES vs. IUPAC)"}
    )


class T5IUPACTokenizer(T5Tokenizer):
    def prepare_for_tokenization(self, text, is_split_into_words=False,
                                 **kwargs):
        return re.sub(" ", "_", text), kwargs

    def _decode(self, *args, **kwargs):
        # replace "_" with " ", except for the _ in extra_id_#
        text = super()._decode(*args, **kwargs)
        text = re.sub("extra_id_", "extraAidA", text)
        text = re.sub("_", " ", text)
        text = re.sub("extraAidA", "extra_id_", text)
        return text

    def sentinels(self, sentinel_ids):
        return self.vocab_size - sentinel_ids - 1

    def sentinel_mask(self, ids):
        return ((self.vocab_size - self._extra_ids <= ids) &
                (ids < self.vocab_size))

    def _tokenize(self, text, sample=False):
        #pieces = super()._tokenize(text, sample=sample)
        pieces = super()._tokenize(text)
        # sentencepiece adds a non-printing token at the start. Remove it
        return pieces[1:]

class T5SMILESTokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i)
                                         for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in x),
                                          additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens "
                    "({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the "
                    "extra_ids tokens"
                )

        super(T5Tokenizer, self).__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        with open(self.vocab_file, "r") as f:
            self.vocab = {}
            vocab_dict = list(map(str.strip, f.readlines()))
            for i in vocab_dict:
                self.vocab[i]=1
            self.vocab = list(self.vocab.keys())
        self.reverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def sentinels(self, sentinel_ids):
        return self.vocab_size - sentinel_ids - 1

    def sentinel_mask(self, ids):
        return ((self.vocab_size - self._extra_ids <= ids) &
                (ids < self.vocab_size))


    @property
    def vocab_size(self):
        return len(self.vocab) + self._extra_ids

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, d):
        self.__dict__ = d

    def _tokenize(self, text):
        tokens = []
        i = 0
        in_brackets = False
        while i < len(text):
            if text[i] in ["[", "<"]:
                in_brackets = True
                tokens.append("")

            if in_brackets:
                tokens[-1] += text[i]
            else:
                if text[i] in ["r", "l"]:
                    # handle Cl & Br
                    tokens[-1] += text[i]
                else:
                    tokens.append(text[i])

            if text[i] in ["]", ">"]:
                in_brackets = False
            i += 1
        return tokens

    def _convert_token_to_id(self, token):
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        else:
            if token in self.reverse_vocab:
                pass
            else:
                print("this token is not in dict:",token)

            return self.reverse_vocab[token]

    def _convert_id_to_token(self, index):
        if index < len(self.vocab):
            token = self.vocab[index]
        else:
            token = "<extra_id_{}>".format(self.vocab_size - 1 - index)
        return token

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix):
        raise NotImplementedError()


@dataclass
class T5Collator:
    pad_token_id: int

    def __call__(self, records):
        # records is a list of dicts
        batch = {}
        padvals = {"input_ids": self.pad_token_id,
                    "smiles_ids":self.pad_token_id,
                   "attention_mask": 0,
                   "labels": -100}
        for k in records[0]:
            if k in padvals:
                batch[k] = pad_sequence([torch.tensor(r[k]) for r in records], #r[k].clone().detach()
                                        batch_first=True,
                                        padding_value=padvals[k])
            else:
                batch[k] = torch.tensor([r[k] for r in records])
        return batch

def label_smoother(model_output, labels):
    epsilon: float = 0.1
    ignore_index: int = -100

    logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    log_probs = -nn.functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    padding_mask = labels.eq(ignore_index)
    # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    # works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss

def prepare_input(data,device):
        """
        Prepares one :obj:`data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        from collections.abc import Mapping
        if isinstance(data, Mapping):
            return type(data)({k: prepare_input(v,device) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(prepare_input(v,device) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=device)
            if data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=data.dtype()))
            return data.to(**kwargs)
        return data

def main():
    torch.manual_seed(42)

    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser((TrainingArguments,
                               DatasetArguments,
                               ModelArguments))
    training_args, dataset_args, model_args = parser.parse_args_into_dataclasses()

    smile_tokenizer_class = T5SMILESTokenizer
    iupac_tokenizer_class = T5IUPACTokenizer

    smile_tokenizer = smile_tokenizer_class(vocab_file=dataset_args.smile_vocab_fn)
    iupac_tokenizer = iupac_tokenizer_class(vocab_file=dataset_args.iupac_vocab_fn)

    # this hack is needed because huggingface doesn't make the tokenizer's
    # special tokens actually special even if you pass them as
    # additional_special_tokens to the tokenizer's __init__
    # (see https://github.com/huggingface/transformers/issues/8999)
    iupac_vocab_size = iupac_tokenizer.vocab_size
    iupac_tokenizer.add_tokens(["<extra_id_{}>".format(i) for i in range(100)],
                         special_tokens=True)

    msg = "extra_ids should already be in vocab"
    assert iupac_tokenizer.vocab_size == iupac_vocab_size, msg

    smile_vocab_size = smile_tokenizer.vocab_size
    smile_tokenizer.add_tokens(["<extra_id_{}>".format(i) for i in range(100)],
                         special_tokens=True)

    msg = "extra_ids should already be in vocab"
    assert smile_tokenizer.vocab_size == smile_vocab_size, msg    

    tokenizer = iupac_tokenizer    

    is_train = 0   # 1 training 0 finetuning

    if is_train:
        torch.save(tokenizer, pt.join("./","real_iupac_tokenizer.pt"))
        torch.save(smile_tokenizer, pt.join("./","real_smile_tokenizer.pt"))
        print("training...",len(tokenizer),len(smile_tokenizer))

    else:
        tokenizer = torch.load(pt.join("./","real_iupac_tokenizer.pt"), map_location="cpu")
        smile_tokenizer = torch.load(pt.join("./","real_smile_tokenizer.pt"), map_location="cpu")

        print('fina_tune...',len(tokenizer),len(smile_tokenizer))

    smile_PAD_IDX = smile_tokenizer.pad_token_id

    if model_args.model_path is None:
        # t5-large uses these params:
        # d_model=1024,
        # d_ff=4096,
        # num_layers=24,
        # num_heads=16,
        config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
        model = T5ForConditionalGeneration(config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_path)

    D = 0
    for p in model.parameters():
        D += p.data.numel()
    print("model dim:", D)

    if model_args.model_path in ["t5-small", "t5-base", "t5-large",
                                 "t5-3B", "t5-11B"]:
        # if we're starting with a model pretrained on natural language,
        # we need to truncate the vocab to our much smaller vocab.
        # but first, we need to move the embeddings for
        # sentinel tokens so they don't get truncated
        old = model.get_input_embeddings().weight.data

        # the extra_ids are not actually at the end of `old` --
        # there are unused embeddings after (maybe for alignment?)
        # get the actual size by tokenizing <extra_id_0> (the last token)
        pretrained_tok = T5Tokenizer.from_pretrained(model_args.model_path)
        old_size = pretrained_tok._convert_token_to_id("<extra_id_0>") + 1
        old = old[:old_size]

        embedding_dim = old.size()[1]
        new_size = tokenizer.vocab_size
        num_extras = tokenizer._extra_ids
        new = torch.cat([old[:new_size - num_extras],
                         old[-num_extras:]], dim=0)
        assert list(new.size()) == [new_size, embedding_dim]
        new_embeddings = torch.nn.Embedding(num_embeddings=new_size,
                                            embedding_dim=embedding_dim,
                                            _weight=new)
        model.set_input_embeddings(new_embeddings)
        model.tie_weights()


    if is_train:
        pass
    else:
        # load weights from checkpoint
        model_fn = os.path.join('./', "9_iupac2iupac_model.pt")
        state_dict = torch.load(model_fn, map_location="cpu")
        model.load_state_dict(state_dict)


    dataset_kwargs = {
            "dataset_dir": dataset_args.dataset_dir,
            "dataset_filename": dataset_args.dataset_filename,
            "tokenizer": tokenizer,
            "smile_tokenizer":smile_tokenizer,
            "max_length": MAXLEN,
            "prepend_target": dataset_args.prepend_target,#控制是否将属性进行高中低的编码放入iupac的编码前
            "target_col": dataset_args.target_col,
            "name_col": dataset_args.name_col,
            "dataset_filename": dataset_args.dataset_filename,
            "low_cutoff": dataset_args.low_cutoff,
            "high_cutoff": dataset_args.high_cutoff,
            "mask_probability": dataset_args.mask_probability,
            "mean_span_length": dataset_args.mean_span_length,
            "smile_name_col":"smiles",
            "return_target":True,#如果设置为否，那么就是label是input_id中的缺失片段，并以此label为预测目标，
            #并且属性已经以高中低编码在了input_id中了，
            #如果设置为真，那么就是返回label 就是具体的属性值, 
            #input_id是iupac的编码，smile_id是smiles的编码id
    }

    if is_train:
        dataset_size=50000
    else:
        dataset_size=90

    train_dataset = IUPACDataset(train=True, **dataset_kwargs)
    eval_dataset = IUPACDataset(train=False, dataset_size=dataset_size,
                                **dataset_kwargs)
    #train_dataset = {
    #                "input_ids": input_ids,
    #                "smiles_ids": smiles_ids
    #                "attention_mask": attention_mask,
    #                "labels": target_ids,                
    #                }
    collator = T5Collator(tokenizer.pad_token_id)  #进行seq对齐 填充padding

    # Prepare optimizer and schedule (linear warmup and sqrt decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
            "params": [p for n, p in model.named_parameters()
                         if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        }, {
            "params": [p for n, p in model.named_parameters()
                         if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=0.0001,
                      eps=training_args.adam_epsilon)

    learning_rate=0.0001

    def lr_lambda(current_step):
        #warmup = training_args.warmup_steps
        warmup = 100
        linear = current_step / warmup**1.5
        sqrt = 1 / (max(warmup, current_step))**0.5
        return learning_rate * min(linear, sqrt)

    lr_schedule = LambdaLR(optimizer, lr_lambda)
    #lr_schedule = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:0.95**epoch)

    n_layers = 6
    n_heads = 4
    model_depth = 512 #512  32128
    ff_depth = 1024
    dropout = 0.1
    N_EPOCHS = 10 #20
    CLIP = 1
    max_length = [tokenizer.vocab_size,smile_tokenizer.vocab_size]

    device = "cuda" if torch.cuda.is_available() else 'cpu' 
    #device = 'cpu'
    print("device:",device,max_length,smile_tokenizer._convert_token_to_id(smile_tokenizer.unk_token))

    iupac2smile_model = Model(max_length, n_layers, n_heads, model_depth, ff_depth, dropout,device).to(device)
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    iupac2smile_model.apply(initialize_weights)

    if is_train:
        pass
    else:
        # load weights from checkpoint
        model_fn = os.path.join('./', "9_iupac2smiles_model.pt")
        state_dict = torch.load(model_fn, map_location="cpu")
        iupac2smile_model.load_state_dict(state_dict)

    #optimizer_iupac2smile = optim.Adam(iupac2smile_model.parameters()+optimizer_grouped_parameters, lr=5e-5)
    optimizer_iupac2smile = optim.Adam([p for n, p in iupac2smile_model.named_parameters()]+[p for n, p in model.named_parameters()], lr=5e-5)
    
    #we ignore the loss whenever the target token is a padding token.
    criterion = nn.CrossEntropyLoss(ignore_index = smile_PAD_IDX)

    #model_path=model_args.model_path
    print("iupac2smile_model smile_PAD_IDX,pad_token_id:",smile_PAD_IDX,tokenizer.pad_token_id)

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=32, #4
            collate_fn=collator,
            shuffle=True)

    eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=32, #4
            collate_fn=collator,
            shuffle=True)

    '''
    # training
    input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

    # inference
    input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # studies have shown that owning a dog is good for you.
    '''

    #model.is_decoder=False
    model.to(device)
    

    loss_vals = []
    loss_vals_eval = []
    save_pt_freq = 1

    train_after = 1

    for epoch in range(N_EPOCHS):
        model.train()
        iupac2smile_model.train()
        
        epoch_loss= []
        pbar = tqdm(train_dataloader)
        pbar.set_description("[Train Epoch {}]".format(epoch)) 
        for  inputs in pbar:
            inputs = prepare_input(inputs,device)
            src = Variable(inputs["input_ids"].to(device))

            #print('iupac id:',*src)

            outputs = model(input_ids=src[:,:-1],decoder_input_ids=src[:,1:],return_dict=True)

            #['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
            #print(outputs.keys(),inputs["input_ids"].shape,labels.shape) #torch.Size([64, 80, 32128]) torch.Size([64, 19])
            #print(outputs.keys()) ['logits', 'past_key_values', 'encoder_last_hidden_state']
            #print(outputs["logits"].shape)

            if outputs["logits"] is not None:
                tr_loss_step = label_smoother(outputs, src[:,1:]) #outputs["logits"]
            else:
                tr_loss_step = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            optimizer.zero_grad()
            tr_loss_step.backward()

            if (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                continue
            #print("iupac2iupac:",tr_loss_step.item())
            pbar.set_postfix(loss=tr_loss_step.item())
            optimizer.step()

            if (epoch+1)>=train_after:        
                trg, src_smile = Variable(inputs["smiles_ids"].to(device)), Variable(outputs["logits"].to(device))
                iupac2smile_model.zero_grad()
                output = iupac2smile_model(src_smile,src[:,1:], trg[:,:-1])
                #trg = [batch size, trg len]
                #output = [batch size, trg len-1, output dim]        
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)  

                #trg = [(trg len - 1) * batch size]
                #output = [(trg len - 1) * batch size, output dim]     
                loss = criterion(output, trg) 
                optimizer_iupac2smile.zero_grad()   
                loss.backward()
                torch.nn.utils.clip_grad_norm_(iupac2smile_model.parameters(), CLIP)
                epoch_loss.append(loss.item())
                optimizer_iupac2smile.step()
                pbar.set_postfix(loss=loss.item())

        lr_schedule.step()
        loss_vals.append(np.mean(epoch_loss))

        if (epoch+1)%save_pt_freq ==0:
            if is_train:
                torch.save(model.state_dict(), str(epoch)+'_iupac2iupac_model.pt')
                print("save iupac2iupac_model:",str(epoch+1)+'_iupac2iupac_model.pt')

                torch.save(iupac2smile_model.state_dict(), str(epoch)+'_iupac2smiles_model.pt')
                print("save iupac2smile_model:",str(epoch+1)+'_iupac2smiles_model.pt')
            else:
                torch.save(model.state_dict(), str(epoch)+'_iupac2iupac_model_fine_tune_non.pt')
                print("save iupac2iupac_model:",str(epoch+1)+'_iupac2iupac_model_fine_tune_non.pt')

                torch.save(iupac2smile_model.state_dict(), str(epoch)+'_iupac2smiles_model_fine_tune_non.pt')
                print("save iupac2smile_model:",str(epoch+1)+'_iupac2smiles_model_fine_tune_non.pt')


        model.eval()
        iupac2smile_model.eval()
        epoch_loss_eval= []
        pbar = tqdm(eval_dataloader)
        pbar.set_description("[Eval Epoch {}]".format(epoch)) 
        for inputs in pbar:
            inputs = prepare_input(inputs,device)

            src_iu = Variable(inputs["input_ids"].to(device))
            outputs = model(input_ids=src_iu[:,:-1],decoder_input_ids=src_iu[:,1:],return_dict=True)

            if outputs["logits"] is not None:
                tr_loss_step = label_smoother(outputs, src_iu[:,1:]) #outputs["logits"]
            else:
                tr_loss_step = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            print("iupac2iupac eval:",tr_loss_step.item(),outputs["logits"].shape,inputs["smiles_ids"].shape,inputs["input_ids"].shape)
            #iupac2iupac eval: 5.413732528686523 torch.Size([64, 62, 32128]) torch.Size([64, 76]) torch.Size([64, 63])

            trg, src_smile = Variable(inputs["smiles_ids"].to(device)), Variable(outputs["logits"].to(device))
            iupac2smile_model.zero_grad()
            output = iupac2smile_model(src_smile,src_iu[:,1:], trg[:,:-1])
            #trg = [batch size, trg len]
            #output = [batch size, trg len-1, output dim]        
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)                     
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]   
            loss = criterion(output, trg)    
            epoch_loss_eval.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        loss_vals_eval.append(np.mean(epoch_loss_eval))    
    if is_train:
        torch.save(model.state_dict(), pt.join("models",'iupac2iupac_model.pt'))
        torch.save(model, pt.join("models","real_iupac2iupac_model.pt"))
        torch.save(iupac2smile_model.state_dict(), pt.join("models","iupac2smiles_model.pt"))
        torch.save(iupac2smile_model, pt.join("models","real_iupac2smiles_model.pt"))
    else:
        torch.save(model.state_dict(), pt.join("models",'iupac2iupac_model_fine_tune_non.pt'))
        torch.save(model, pt.join("models","real_iupac2iupac_model_fine_tune_non.pt"))
        torch.save(iupac2smile_model.state_dict(), pt.join("models","iupac2smiles_model_fine_tune_non.pt"))
        torch.save(iupac2smile_model, pt.join("models","real_iupac2smiles_model_fine_tune_non.pt"))

    print(loss_vals,loss_vals_eval)

if __name__ == "__main__":
    main()



#python t5_real.py --output_dir ./  --dataset_dir ./download_pubchem/ --smile_vocab_fn ./vocab/smile.vocab --iupac_vocab_fn ./vocab/iupac_spm.model --dataset_filename ./iupacs_properties_100.csv
#python t5_real.py --output_dir ./ --dataset_dir ./download_pubchem/ --smile_vocab_fn ./vocab/smile.vocab --iupac_vocab_fn ./vocab/iupac_spm.model --dataset_filename ./pubchem_30m_new.csv


#nohup python t5_real.py --output_dir ./ --dataset_dir ./download_pubchem/ --smile_vocab_fn ./vocab/smile.vocab --iupac_vocab_fn ./vocab/iupac_spm.model --dataset_filename ./pubchem_30m_new.csv > real.txt &

#nohup python t5_real.py --output_dir ./ --dataset_dir ./download_pubchem/ --smile_vocab_fn ./vocab/smile.vocab --iupac_vocab_fn ./vocab/iupac_spm.model --dataset_filename ./SARS0729_canon_desc.csv  > real_fine_tune.txt &

#Preferred|Traditional|Canonical<|Mass|Formula|Log P   iupacs_properties_all.csv

#PUBCHEM_COMPOUND_CID,PUBCHEM_IUPAC_NAME,canon_smiles,aLogP    pubchem_30m.csv -> "PUBCHEM_COMPOUND_CID","Preferred","Canonical<","Log P"

# iupac2iupac2smile  联合一起训练的版本



#python t5_real.py --output_dir ./ --dataset_dir ./download_pubchem/ --smile_vocab_fn ./vocab/smile.vocab --iupac_vocab_fn ./vocab/iupac_spm.model --dataset_filename ./traindata_new.csv


#traindata_new.csv   smiles|aLogP|canonical_smiles|CanonicalSMILES|IUPACName|XLogP

#finetunev1_new.csv  smiles|aLogP|canonical_smiles|CanonicalSMILES|IUPACName|XLogP