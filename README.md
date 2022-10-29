# TransAntivirus

Deep molecular generative model for antiviral drug design with variant Transformer

![Model Architecture of TransAntivirus](https://github.com/AspirinCode/TransAntivirus/blob/main/image/TransAntivirus.png)

## Acknowledgements
We thank the authors of C5T5: Controllable Generation of Organic Molecules with Transformers and IUPAC2Struct: Transformer-based artificial neural networks for the conversion between chemical notationsfor releasing their code. The code in this repository is based on their source code release (https://github.com/dhroth/c5t5 and https://github.com/sergsb/IUPAC2Struct). 

If you find this code useful, please consider citing their work.


## Requirements
```python
Python==3.6
pandas==1.1.5
numpy==1.19.2
pytorch==1.10.0
pytorch-mutex==1.0
torchaudio==0.10.0
torchtext==0.11.2
torchvision==0.11.1
RDKit==2020.03.3.0
transformers==4.18.0
```

https://github.com/rdkit/rdkit



## Model



For the generation stage the model files are available. It is possible to use the ones that are generated during the training step or you can download the ones that we have already generated model files from Google Drive. 




## Generation
novel compound generation:

```python
#traindata_new.csv   smiles|aLogP|IUPACName

#finetunev1_new.csv  smiles|aLogP|IUPACName

python gen_t5_real.py  --dataset_dir ./download_pubchem/ --vocab_fn ./vocab/iupac_spm.model --dataset_filename ./finetunev1_new.csv  > gen_real_fine_tune_non.txt
```

## Model Metrics
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI



## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


## Cite:

