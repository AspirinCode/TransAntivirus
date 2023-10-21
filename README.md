[![License: GNU](https://img.shields.io/badge/License-GNU-yellow)](https://github.com/AspirinCode/TransAntivirus)
[![JCIM](https://img.shields.io/badge/10.1021%2Facs.jcim.3c00536-green)](https://doi.org/10.1021/acs.jcim.3c00536)

# TransAntivirus

**Transformer-based molecular generative model for antiviral drug design**

![Model Architecture of TransAntivirus](https://github.com/AspirinCode/TransAntivirus/blob/main/image/TransAntivirus.png)

## Acknowledgements
We thank the authors of C5T5: Controllable Generation of Organic Molecules with Transformers and IUPAC2Struct: Transformer-based artificial neural networks for the conversion between chemical notations for releasing their code. The code in this repository is based on their source code release (https://github.com/dhroth/c5t5 and https://github.com/sergsb/IUPAC2Struct). If you find this code useful, please consider citing their work.


### Codes & Model of Hugging Face

https://huggingface.co/superspider2023/TransAntivirus/tree/main


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



## Model & data



For the generation stage the model files are available. It is possible to use the ones that are generated during the training step or you can download the ones that we have already generated model files from Google Drive. 

https://drive.google.com/drive/u/0/folders/1T2CuAo52Auryepr9UZOSB1g6U_i332UY


## Generation
novel compound generation:

```python
#traindata_new.csv   smiles|aLogP|IUPACName

#finetunev1_new.csv  smiles|aLogP|IUPACName

python gen_t5_real.py  --dataset_dir ./download_pubchem/ --vocab_fn ./vocab/iupac_spm.model --dataset_filename ./finetunev1_new.csv  > gen_real_fine_tune_non.txt
```

## Model Metrics
### MOSES
Molecular Sets (MOSES), a benchmarking platform to support research on machine learning for drug discovery. MOSES implements several popular molecular generation models and provides a set of metrics to evaluate the quality and diversity of generated molecules. With MOSES, MOSES aim to standardize the research on molecular generation and facilitate the sharing and comparison of new models.
https://github.com/molecularsets/moses


### QEPPI
quantitative estimate of protein-protein interaction targeting drug-likeness

https://github.com/ohuelab/QEPPI

*  Kosugi T, Ohue M. Quantitative estimate index for early-stage screening of compounds targeting protein-protein interactions. International Journal of Molecular Sciences, 22(20): 10925, 2021. doi: 10.3390/ijms222010925
Another QEPPI publication (conference paper)

*  Kosugi T, Ohue M. Quantitative estimate of protein-protein interaction targeting drug-likeness. In Proceedings of The 18th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB 2021), 2021. doi: 10.1109/CIBCB49929.2021.9562931 (PDF) * © 2021 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.


## License
Code is released under GNU AFFERO GENERAL PUBLIC LICENSE.


## Cite:

*  Jiashun Mao, Jianmin Wang, Amir Zeb, Kwang-Hwi Cho, Haiyan Jin, Jongwan Kim, Onju Lee, Yunyun Wang, and Kyoung Tai No. "Transformer-Based Molecular Generative Model for Antiviral Drug Design" Journal of Chemical Information and Modeling, 2023;, [DOI: 10.1021/acs.jcim.3c00536](https://doi.org/10.1021/acs.jcim.3c00536)

*  Rothchild, Daniel, Alex Tamkin, Julie Yu, Ujval Misra, and Joseph Gonzalez. "C5t5: Controllable generation of organic molecules with transformers." arXiv preprint arXiv:2108.10307 (2021).

*  Jianmin Wang, Yanyi Chu, Jiashun Mao, Hyeon-Nae Jeon, Haiyan Jin, Amir Zeb, Yuil Jang, Kwang-Hwi Cho, Tao Song, Kyoung Tai No, De novo molecular design with deep molecular generative models for PPI inhibitors, Briefings in Bioinformatics, 2022;, bbac285, https://doi.org/10.1093/bib/bbac285
