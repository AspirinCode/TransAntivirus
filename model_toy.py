import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformer_toy  import Transformer
def subsequent_mask (tgt_mask):
    size = tgt_mask.size(-1)
    return tgt_mask.to(torch.uint8) & torch.tril(torch.ones(1,size,size, dtype=torch.uint8)).to(tgt_mask.device)

class DataParallel (torch.nn.Module):
    def __init__(self, model):
        super(DataParallel, self).__init__()
        self.model = torch.nn.DataParallel(model).cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

class Model(nn.Module):
    def __init__ (self, vocab_size, n_layers, n_heads, model_depth, ff_depth, dropout,device):
        super(Model, self).__init__()
        self.T = Transformer(vocab_size, n_layers, n_heads, model_depth, ff_depth, dropout)
        self.device = device
        self.w1 = nn.Linear(32128,model_depth)

    def forward(self, src_em,src, tgt):
        #src = pad_pack(src).to(self.device)
        src_mask = (src != 0).unsqueeze(-2).to(self.device)
        #tgt = pad_pack(tgt).to(self.device)
        tgt_mask = subsequent_mask((tgt != 0).unsqueeze(-2)).type_as(src_mask)
        out = self.T(self.w1(src_em), tgt, src_mask, tgt_mask)
        
        #out = self.T.generator(out)[:,-1]
        out = self.T.generator(out)

        return out

    def predict_single (self, inp,x, beam=1, temp=0, naug=0):
        naug = 0
        naug += 1
        with torch.no_grad():
            self.T.eval()

            src = []
            src_x = []
            for b in range(beam):
                src += [inp]
                src_x+=[x]

            src = pad_pack_float(src).to(self.device)   
            src = self.w1(src)

            src_x = pad_pack(src_x).to(self.device)

            src_mask = (src_x != 0).unsqueeze(-2).to(self.device)

            #print(src_mask.shape)
            #print(src.shape)            

            src_mem = self.T.encoder(src, src_mask)

            cands = [[766] for b in range(beam)]
            probs = [[] for b in range(beam)]
            res_probs = []
            res = []

            #print(src_mem.shape) 


            for step in range(150):
                if beam <= 0:
                    break

                aug_cands = []
                for c in cands:
                    aug_cands += [c]*naug

                tgt = pad_pack(aug_cands).to(self.device)

                tgt_mask = subsequent_mask((tgt != 0).unsqueeze(-2)).type_as(src_mask)
                out = self.T.decoder(self.T.tgt_embedder(tgt), src_mem, src_mask, tgt_mask)
                out = self.T.generator(out)[:,-1]

                out = out.view(beam, naug, out.shape[-1])
                out = out.mean(dim=1)

                pbs,wds = torch.sort(out, dim=1, descending=True)
                step_cands = []
                step_probs = []
                for i in range(beam):
                    for j in range(beam*2):
                        step_cands.append(cands[i]+[wds[i,j].tolist()])
                        step_probs.append(probs[i]+[pbs[i,j].tolist()])
                step_cands, step_probs = remove_duplicates(step_cands, step_probs)
                best_ids = np.argsort([prob_score(pb) for pb in step_probs])[::-1][:beam]

                cands = []
                probs = []
                for i in best_ids:
                    if step_cands[i][-1] == 1:
                        res.append(step_cands[i])
                        res_probs.append(step_probs[i])
                        beam -= 1
                        src_mem = src_mem[:-1*naug]
                        src_mask = src_mask[:-1*naug]
                        src = src[:-1*naug]
                    else:
                        cands.append(step_cands[i])
                        probs.append(step_probs[i])

        if beam > 0:
            res += cands[:beam]
            res_probs += probs[:beam]


        pred = []
        
        for r in res:
            try:
                def is_digit(str):
                    try:
                        tmp = float(str)
                        return True
                    except ValueError:
                        return False

                pred.append(r)
                
            except Exception as inst:
                print("error:",inst)
                pred.append(None)

        pred_probs = [prob_score(pb) for pb in res_probs]
        order = np.argsort(pred_probs)[::-1]
        final_outputs = np.array(pred)[order].tolist()
        final_probs = np.array(pred_probs)[order].tolist()
        return final_outputs, final_probs

    def predict (self, X,src, beam=1, temp=0, naug=0):
        pred = [[None for i in range(len(X))] for b in range(beam)]
        for i in tqdm(range(len(X))):
            cands, probs = self.predict_single(X[i],src[i], beam=beam, temp=temp, naug=naug)
            if len(cands) > 0:
                for b in range(beam):
                    pred[b][i] = cands[b]
        return pred

def prob_score (prob):
    return np.exp(np.mean(prob))

def remove_duplicates (cands, probs):
    use_cands = []
    use_probs = []
    for i in range(len(cands)):
        if cands[i] not in use_cands:
            use_cands.append(cands[i])
            use_probs.append(probs[i])
    return use_cands, use_probs

def pad_pack (sequences):
    maxlen = max(map(len, sequences))
    batch = torch.LongTensor(len(sequences),maxlen).fill_(0)
    for i,x in enumerate(sequences):
        batch[i,:len(x)] = torch.LongTensor(x)
    return batch

def pad_pack_float(sequences):
    maxlen = max(map(len, sequences))
    batch = torch.FloatTensor(len(sequences),maxlen,len(sequences[0][0])).fill_(0.0)
    for i,x in enumerate(sequences):
        batch[i,:len(x)] = torch.FloatTensor(x)
    return batch