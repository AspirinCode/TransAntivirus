from transformers import (
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
#from t5 import T5IUPACTokenizer, T5SMILESTokenizer, T5Collator
from iupac_dataset import IUPACDataset
from data_utils import collapse_sentinels

from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import copy
import itertools
import operator
import math
import random
import math
import sys
import os.path as pt
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from model_toy import Model
MAXLEN = 128
from t5_real import T5IUPACTokenizer,T5SMILESTokenizer,T5Collator
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# set this to 0 unless using a model pretrained without overriding
# _tokenize in T5IUPACTokenizer (hack to deal with old pretrained models)
H = 0

@dataclass
class IUPACArguments:
    dataset_dir: str = field(
            metadata={"help": "Directory where dataset is locaed"}
    )
    vocab_fn: str = field(
            metadata={"help": "File containing sentencepiece model"}
    )
    dataset_filename: str = field(
            default="iupacs_logp.txt",
            metadata={"help": "Filename within dataset_dir containing the data"}
    )
    model_path: Optional[str] = field(
            default=None,
            metadata={"help": "Checkpoint to use"}
    )
    tokenizer_type: Optional[str] = field(
            default="IUPAC",
            metadata={"help": "How to tokenize chemicals (SMILES vs. IUPAC)"}
    )
    target_col: Optional[str] = field(
            default="aLogP",
            metadata={"help": "Name of column with target property values"}
    )
    low_cutoff: Optional[float] = field(
            default=-0.4, # for logp
            metadata={"help": "Cutoff between <low> and <med> tokens"}
    )
    high_cutoff: Optional[float] = field(
            default=5.6, # for logp
            metadata={"help": "Cutoff between <med> and <high> tokens"}
    )
    name_col: Optional[str] = field(
            default="IUPACName",
            metadata={"help": "Name of column with IUPAC names"}
    )
    conversion_pairs: Optional[str] = field(
            default="all_all",
            metadata={"help": "high_low means molecules with ground truth <low> " +
                              "will be generated with <high>, and vice versa. " +
                              "all_all means all molecules will be generated " +
                              "with all of <low>, <med>, and <high>"}
    )
    num_orig_iupacs: Optional[int] = field(
            default=1000,
            metadata={"help": "how many starting molecules to generate from"}
    )
    masks_per_iupac: Optional[int] = field(
            default=-1,
            metadata={"help": "how many masks to use per source molecule (-1=all)"}
    )
    balanced_sample: Optional[bool] = field(
            default=False,
            metadata={"help": "Use an equal number of source iupacs per tgt val"}
    )


def mask_name(inputs, masked_ids, tokenizer):
    orig_inputs = inputs
    inputs = inputs.clone()

    masked_ids = torch.tensor(masked_ids)
    mask = torch.zeros_like(inputs).bool()
    mask[masked_ids] = True
    inputs[mask] = -1
    inputs[~mask] = torch.arange(inputs.numel())[~mask]
    inputs = torch.unique_consecutive(inputs)
    mask = inputs == -1
    inputs[~mask] = orig_inputs[inputs[~mask]]
    inputs[mask] = tokenizer.sentinels(torch.arange(mask.sum()))
    return inputs


def generate(model,iupac2smile_model, tokenizer,smile_tokenizer, inputs_list, masked_indices,smiles_ids, n_candidates=1):
    if not isinstance(masked_indices[0], (list, tuple)):
        raise ValueError("must supply a list of masks")

    # inputs_list is a 1D list of tensors
    # masked_indices is a 3D list
    print('input IUPAC name',[tokenizer.decode(i[1:-1]) for i in inputs_list])
    print('input SMILES name',[smile_tokenizer.decode(i[1:-1]) for i in smiles_ids])

    batch = []
    split_sizes = []
    for inputs, ms in zip(inputs_list, masked_indices):
        orig_inputs = inputs.clone()

        # add to batch, where each element is inputs with a different mask
        for m in ms:
            batch.append(mask_name(inputs, m, tokenizer).cuda())
        split_sizes.append(len(ms) * n_candidates)

    pad = tokenizer.pad_token_id
    print('iupac pad id,eos is end mark',pad,tokenizer.eos_token_id)  #0 1  eos is end mark
    model.eval()
    iupac2smile_model.eval()

    minibatches = itertools.zip_longest(*[iter(batch)] * 16,
                                        fillvalue=torch.tensor([]))
    count = math.ceil((len(batch) - 1) / 16)
    number = 1

    minibatch_gen = []
    for minibatch in minibatches:
        minibatch = pad_sequence(minibatch,
                                 batch_first=True,
                                 padding_value=pad)

        #print("minibatch:",*minibatch,minibatch.shape) #minibatch: torch.Size([16, 29])

        print("\ncurrent number:",number,'total:',count)

        for i in minibatch:
            print(tokenizer.decode(i[1:-1]))


        outputs = model(input_ids=minibatch[:,:-1],decoder_input_ids=minibatch[:,1:],return_dict=True)
        outputs = outputs["logits"].cpu()
        #print('outputs["logits"]:',outputs.shape)

        smile_batch = iupac2smile_model.predict(outputs,minibatch[:,1:].cpu())
        smile_batch = smile_batch[0]
        #print("smile_batch:",*smile_batch) #[766, 66, 23, 5, 17, 23, 23, 63, 66, 64, 63, 23, 22, 6, 22, 22, 22, 27, 22, 6, 64, 66, 66, 5, 1]

        print('output smiles:')

        for i in smile_batch:
            print(smile_tokenizer.decode(i[1:-1]))

        #print(tokenizer.unk_token_id,smile_tokenizer.unk_token_id) #2 766  start mark

        number = number + 1

        smile_batch = [torch.tensor(j) for j in smile_batch]

        #smile_batch = pad_sequence(smile_batch,batch_first=True,padding_value=pad)

        minibatch_gen.append(smile_batch)

    # truncate last minibatch to correct size
    last_minibatch_size = (len(batch) - 1) % 16 + 1

    print('last_minibatch_size:',len(batch),last_minibatch_size) #last_minibatch_size: 1820 12

    minibatch_gen[-1] = minibatch_gen[-1][:last_minibatch_size]  

    minibatch_gen = [pad_sequence(i,batch_first=True,padding_value=pad) for i in minibatch_gen]

    assert sum(m.size(0) for m in minibatch_gen) == len(batch)

    max_len = max([g.size(1) for g in minibatch_gen])

    print([g.size(1) for g in minibatch_gen],max_len)

    padded = [torch.cat([g, pad * torch.ones(g.size(0),
                                             max_len - g.size(1)).long().to(g.device)],
                          dim=1)
              if g.size(1) < max_len else g
              for g in minibatch_gen]
    generated = torch.cat(padded, dim=0)

    generated_split = generated.split(split_sizes)

    def remove_extraneous(ids):
        # delete everything after </s>
        eos = smile_tokenizer.eos_token_id
        pad_mask = (ids == eos).cumsum(dim=0).clamp(0, 1).bool()
        ids = ids[:ids.numel() - pad_mask.sum()]
        return ids    

    all_interleaved = []
    for generated, orig in zip(generated_split, smiles_ids):
        interleaved = {}
        n_invalid = 0
        for i in range(generated.shape[0]):
            try:
                #  delete everything after </s>
                gen = remove_extraneous(generated[i])            
                decoded = smile_tokenizer.decode(gen[1:])

                is_dup = decoded in interleaved
                is_shorter = False
                if is_dup:
                    is_shorter = gen[1:].numel() < interleaved[decoded].numel()
                if not is_dup or (is_dup and is_shorter):
                    interleaved[decoded] = gen[1:].cpu()
            except ValueError:
                n_invalid += 1

        interleaved = [(decoded,
                        levenshtein_distance(orig[1:-1], tokens))
                       for decoded, tokens in interleaved.items()]
        all_interleaved.append(interleaved)

    return all_interleaved

def mask_ids(length, span_lengths):
    max_id = length - span_lengths[-1] + 1
    comb = itertools.combinations(range(2, max_id),
                                  len(span_lengths))
    sli = range(len(span_lengths))
    masked = []
    for c in comb:
        new = list(itertools.chain(
                  *[range(start, start + slen)
                    for i, start, slen in zip(sli, c, span_lengths)
                   ]
              ))
        # check that it's actually len(span_lengths) spans
        nbreaks = sum([new[i+1] > new[i] + 1 for i in range(len(new) - 1)])
        if nbreaks == len(span_lengths) - 1:
            masked.append(new)
    return masked

masks_cache = {}
def get_masks(length, span_lengths):
    key = (length, tuple(map(tuple, span_lengths)))
    if key in masks_cache:
        return masks_cache[key]
    else:
        masks = [(sl, m) for sl in span_lengths for m in mask_ids(length, sl)]
        masks_cache[key] = masks
        return masks

def main():
    torch.manual_seed(42)

    parser = HfArgumentParser(IUPACArguments)
    iupac_args, = parser.parse_args_into_dataclasses()

    # get the list of molecules to generate from
    '''
    if iupac_args.tokenizer_type == "IUPAC":
        tokenizer_class = T5IUPACTokenizer
    elif iupac_args.tokenizer_type == "SMILES":
        tokenizer_class = T5SMILESTokenizer
    else:
        msg = "Unsupported tokenization type {}"
        raise RuntimeError(msg.format(iupac_args.tokenizer_type))

    tokenizer = tokenizer_class(vocab_file=iupac_args.vocab_fn)

    smile_tokenizer_class = T5SMILESTokenizer

    smile_vocab_fn ='./vocab/smile.vocab'
    smile_tokenizer = smile_tokenizer_class(vocab_file=smile_vocab_fn)

    iupac_vocab_size = tokenizer.vocab_size
    tokenizer.add_tokens(["<extra_id_{}>".format(i) for i in range(100)],
                         special_tokens=True)

    msg = "extra_ids should already be in vocab"
    assert tokenizer.vocab_size == iupac_vocab_size, msg

    smile_vocab_size = smile_tokenizer.vocab_size
    smile_tokenizer.add_tokens(["<extra_id_{}>".format(i) for i in range(100)],
                         special_tokens=True)

    msg = "extra_ids should already be in vocab"
    assert smile_tokenizer.vocab_size == smile_vocab_size, msg

    smile_PAD_IDX = smile_tokenizer.pad_token_id

    print(len(tokenizer),len(smile_tokenizer))
    '''
    tokenizer = torch.load(pt.join("./","real_iupac_tokenizer.pt"), map_location="cpu")
    smile_tokenizer = torch.load(pt.join("./","real_smile_tokenizer.pt"), map_location="cpu")

    print(len(tokenizer),len(smile_tokenizer))

    ########################################################################################################
    is_train = 0  # 1 training 模式下的生成  0  finetuning模式下的生成

    if is_train:
        dataset_size =14000000
    else:
        dataset_size = 70

    dataset_kwargs = {
            "dataset_dir": iupac_args.dataset_dir,
            "tokenizer": tokenizer,
            "smile_tokenizer":smile_tokenizer,
            "max_length": MAXLEN,
            "prepend_target": True,
            "low_cutoff": iupac_args.low_cutoff,
            "high_cutoff": iupac_args.high_cutoff,
            "target_col": iupac_args.target_col,
            "name_col": iupac_args.name_col,
            "dataset_size": dataset_size,
            "mean_span_length": 3,
            "mask_probability": 0,
            "dataset_filename": iupac_args.dataset_filename,
            "smile_name_col":"smiles",
            "return_target":True
    }
    eval_dataset = IUPACDataset(train=False, **dataset_kwargs)

#####################################################################################################
    # get the trained model
    #config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
    #model = T5ForConditionalGeneration(config)

    if iupac_args.model_path is None:
        # t5-large uses these params:
        # d_model=1024,
        # d_ff=4096,
        # num_layers=24,
        # num_heads=16,
        config = T5Config(decoder_start_token_id=tokenizer.pad_token_id)
        model = T5ForConditionalGeneration(config)
    else:
        model = T5ForConditionalGeneration.from_pretrained(iupac_args.model_path)

    D = 0
    for p in model.parameters():
        D += p.data.numel()
    print("model dim:", D)

    if iupac_args.model_path in ["t5-small", "t5-base", "t5-large",
                                 "t5-3B", "t5-11B"]:
        # if we're starting with a model pretrained on natural language,
        # we need to truncate the vocab to our much smaller vocab.
        # but first, we need to move the embeddings for
        # sentinel tokens so they don't get truncated
        old = model.get_input_embeddings().weight.data

        # the extra_ids are not actually at the end of `old` --
        # there are unused embeddings after (maybe for alignment?)
        # get the actual size by tokenizing <extra_id_0> (the last token)
        pretrained_tok = T5Tokenizer.from_pretrained(iupac_args.model_path)
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


    #model.resize_token_embeddings(len(tokenizer))

    # load weights from checkpoint
    if is_train:  
        model_fn = os.path.join('./', "9_iupac2iupac_model.pt")
    else:
        model_fn = os.path.join('./', "19_iupac2iupac_model_fine_tune_non.pt")

    state_dict = torch.load(model_fn, map_location="cpu")
    model.load_state_dict(state_dict)
    model.tie_weights()

    device = "cuda" if torch.cuda.is_available() else 'cpu' 
    #device = 'cpu'
    print("device:",device)

#####################################################################################################
    model.eval()
    model = model.to(device)

    collator = T5Collator(tokenizer.pad_token_id)

    low = tokenizer._convert_token_to_id("<low>")
    med = tokenizer._convert_token_to_id("<med>")
    high = tokenizer._convert_token_to_id("<high>")

    if iupac_args.conversion_pairs == "high_low":
        orig_iupacs = {"low": [], "high": []}
    elif iupac_args.conversion_pairs == "all_all":
        orig_iupacs = {"low": [], "med": [], "high": []}

    if is_train:
        pass
    else:
        iupac_args.num_orig_iupacs = 70

    iupacs_per_key = math.ceil(iupac_args.num_orig_iupacs / len(orig_iupacs.keys()))

    N = iupac_args.num_orig_iupacs
    print('total,num_orig_iupacs:',N)

    i = 0
    while len(list(itertools.chain(*orig_iupacs.values()))) < N:
        input_ids = eval_dataset[i]["input_ids"]
        too_long = input_ids.numel() > 200
        has_unk = (input_ids == tokenizer.unk_token_id).sum() > 0
        if has_unk:  # 空格会被转换为unk，因此直接去掉这些空格即可
            eval_dataset[i]["input_ids"] = eval_dataset[i]["input_ids"][[~np.isin(eval_dataset[i]["input_ids"],torch.tensor([2]))]]
            has_unk = False

        if not has_unk and not too_long:
            first = input_ids[H].item()
            key = {low: "low", med: "med", high: "high"}[first]
            if key in orig_iupacs:
                if iupac_args.balanced_sample:
                    # get iupacs_per_key for each key
                    if len(orig_iupacs[key]) <= iupacs_per_key:
                        orig_iupacs[key].append(eval_dataset[i])
                else:
                    # take every non-unk-containing iupac
                    orig_iupacs[key].append(eval_dataset[i])
        else:
            # ignore names with <unk> in them and very long names
            #pass
            print(i,has_unk,too_long,input_ids)
        i += 1

    assert len(list(itertools.chain(*orig_iupacs.values()))) == N

    generated_iupacs = []
    for datum in itertools.chain(*orig_iupacs.values()):
        inputs = datum["input_ids"]
        smiles_ids = datum["smiles_ids"]

        #span_lengths = [[1], [2], [3], [1, 1], [1, 2], [2, 1], [2, 2]]
        # if you change span_lengths, you need to change the code in
        # best_in_dataset.py too if you want best_in_dataset.py to correctly
        # find molecules that could have been generated by gen_t5.py
        span_lengths = [[1], [2], [3], [4], [5]]

        if iupac_args.conversion_pairs == "high_low":
            # only change from <low> to <high>
            if inputs[H] == low:
                orig_logp = "low"
                new_logps = ["high"]
            elif inputs[H] == high:
                orig_logp = "high"
                new_logps = ["low"]
        elif iupac_args.conversion_pairs == "all_all":
            # try all of <low>, <med> and <high> for all molecules
            orig_logp = {low: "low", med: "med", high: "high"}[inputs[H].item()]
            new_logps = ["low", "med", "high"]

        for new_logp in new_logps:
            inputs[H] = {"low": low, "med": med, "high": high}[new_logp]

            # don't print out <high>/<med>/<low> and </s>
            orig = tokenizer.decode(inputs[1+H:-1])
            orig_smile = smile_tokenizer.decode(smiles_ids[1:-1])

            base_out_dict = {"orig": orig,
                             "orig_smile":orig_smile,
                             "orig_logp": orig_logp,
                             "new_logp": new_logp}

            masks = get_masks(inputs.numel(), span_lengths)

            if iupac_args.masks_per_iupac > -1:
                masks = random.sample(masks, iupac_args.masks_per_iupac)

            # sort by slen and then group by slen
            grouped = itertools.groupby(sorted(masks, key=lambda x:x[0]),
                                        operator.itemgetter(0))

            for slens, group in grouped:
                masks = [t[1] for t in group]

                d = base_out_dict.copy()
                d.update({"nspans": len(slens),
                         "span_lengths": ",".join(map(str, slens)),
                         "gen": (inputs.clone(), masks,smiles_ids.clone())})

                generated_iupacs.append(d)

    n_layers = 6
    n_heads = 4
    model_depth = 512 #512  32128
    ff_depth = 1024
    dropout = 0.1
    N_EPOCHS = 10
    CLIP = 1
    max_length = [tokenizer.vocab_size,smile_tokenizer.vocab_size]

    iupac2smile_model = Model(max_length, n_layers, n_heads, model_depth, ff_depth, dropout,device).to(device)

    # load weights from checkpoint
    if is_train:
        model_fn = os.path.join('./', "9_iupac2smiles_model.pt")
    else:
        model_fn = os.path.join('./', "19_iupac2smiles_model_fine_tune_non.pt")

    state_dict = torch.load(model_fn, map_location="cpu")
    iupac2smile_model.load_state_dict(state_dict)

    iupac2smile_model.eval()
    iupac2smile_model = iupac2smile_model.to(device)


    # actually generate now
    gen = generate(model,iupac2smile_model,
                   tokenizer,smile_tokenizer,
                   [d["gen"][0] for d in generated_iupacs],
                   [d["gen"][1] for d in generated_iupacs],
                   [d["gen"][2] for d in generated_iupacs])

    for i, g in enumerate(gen):
        generated_iupacs[i]["gen"] = g

    # print output
    headers = ["orig","orig_smile", "orig_logp", "new_logp", "nspans", "span_lengths",
               "levenshtein_distance", "generated_smiles"]
    print(",".join(headers))
    unique_smiles = set()
    df_final = []

    for record in generated_iupacs:
        # orig, orig_smile,orig_logp, final_logp, gen, nspans, span_lengths
        try:
            orig_idx = record["gen"].index((record["orig_smile"], 0))
            record["gen"].pop(orig_idx)
        except ValueError:
            # orig not in generated, so no need to remove it
            pass
        for smiles, edit_distance in record["gen"]:
            cols = [record["orig"],record["orig_smile"], record["orig_logp"],
                    record["new_logp"], str(record["nspans"]),
                    record["span_lengths"], str(edit_distance), smiles]
            # check if equal to orig since it's possible to have
            # an edit distance > 0 but tokenize.decode() to the same
            # IUPAC name
            if smiles not in unique_smiles and smiles != record["orig_smile"]:
                unique_smiles.add(smiles)
                print('"' + '","'.join(cols) + '"')

                df_final.append(cols)

    aaa = pd.DataFrame(df_final,columns=headers)
    if is_train:
        aaa.to_csv("df_final_real.csv",sep="|",index= None)
    else:
        aaa.to_csv("df_final_real_fine_tune_non.csv",sep="|",index= None)
    
    print(aaa)

# from https://rosettacode.org/wiki/Levenshtein_distance#Python
def levenshtein_distance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

if __name__ == "__main__":
    main()


#nohup python gen_t5_real.py  --dataset_dir ./download_pubchem/    --vocab_fn ./vocab/iupac_spm.model --dataset_filename ./pubchem_30m_new.csv

#nohup python gen_t5_real.py  --dataset_dir ./download_pubchem/ --vocab_fn ./vocab/iupac_spm.model --dataset_filename ./SARS0729_canon_desc.csv  > gen_real_fine_tune.txt &


#traindata_new.csv   smiles|aLogP|canonical_smiles|CanonicalSMILES|IUPACName|XLogP

#finetunev1_new.csv  smiles|aLogP|canonical_smiles|CanonicalSMILES|IUPACName|XLogP

#nohup python gen_t5_real.py  --dataset_dir ./download_pubchem/ --vocab_fn ./vocab/iupac_spm.model --dataset_filename ./finetunev1_new.csv  > gen_real_fine_tune_non.txt &
