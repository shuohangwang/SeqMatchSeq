# SeqMatchSeq
Implementations of there modes described in the three papers related to sequence mathing:

- [Learning Natural Language Inference with Lstm](https://arxiv.org/abs/1512.08849) by Shuohang Wang, Jing Jiang

- [Machine Comprehension Using Match-LSTM and Answer Pointer](https://arxiv.org/abs/1608.07905) by Shuohang Wang, Jing Jiang

- A Compare-Aggregate Model for Matching Text Sequences by Shuohang Wang, Jing Jiang

# Learning Natural Language Inference with Lstm

### Requirements
- [Torch7](https://github.com/torch/torch7)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- Python 2.7

### Datasets
- [The Stanford Natural Language Inference (SNLI) Corpus](http://nlp.stanford.edu/projects/snli/)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Usage
```
sh preprocess.sh snli
cd main
th main.lua -task snli -model mLSTM -dropoutP 0.3 -num_classes 3
```

`squad_preprocess.sh` will download the datasets and preprocess the SNLI corpus into the files 
(train.txt dev.txt test.txt) under the path "data/snli/sequence" with the format:

>sequence1(premise) \t sequence2(hypothesis) \t label(from 1 to num_classes) \n

`main.lua` will first initialize the prepossed data and word embeddings into a Torch format and 
then run the alogrithm. "dropoutP" is the main prarameter we tuned.

### Docker
You may try to use Docker for running the code.
- [Docker Install](https://github.com/codalab/codalab-worksheets/wiki/Installing-Docker)
- [Image](https://hub.docker.com/r/shuohang/seqmatchseq/): docker pull shuohang/seqmatchseq:1.0

After installation, just run the following codes:
```
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt      shuohang/seqmatchseq:1.0 /bin/bash -c "sh preprocess.sh snli"
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt/main shuohang/seqmatchseq:1.0 /bin/bash -c "th main.lua"
```
# Machine Comprehension Using Match-LSTM and Answer Pointer

### Requirements
- [Torch7](https://github.com/torch/torch7)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- [parallel](https://github.com/clementfarabet/lua---parallel)
- Python 2.7
- Python Packages: [NLTK](http://www.nltk.org/install.html), collections, json, argparse
- [NLTK Data](http://www.nltk.org/data.html): punkt
- Multiple-cores CPU

### Datasets
- [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Usage
```
sh preprocess.sh squad
cd main
th mainDt.lua 
```

`squad_preprocess.sh` will download the datasets and preprocess the SQuAD corpus into the files 
(train.txt dev.txt) under the path "data/squad/sequence" with the format:

>sequence1(Doument) \t sequence2(Question) \t sequence of the positions where the answer appear 
in Document (e.g. 3 4 5 6)  \n

`mainDt.lua` will first initialize the prepossed data and word embeddings into a Torch format and 
then run the alogrithm. As this code is run through multiple CPU cores, the initial parameters are
written in file "main/init.lua". 

- `opt.num_processes`: 5. The number of threads used.
- `opt.batch_size`   : 6. Batch size for each thread. (Then the mini_batch would be 5*6 .)
- `opt.model`        : boundaryMPtr (boundary model); sequenceMPtr (sequence model)

## Docker
You may try to use Docker for running the code.
- [Docker Install](https://github.com/codalab/codalab-worksheets/wiki/Installing-Docker)
- [Image](https://hub.docker.com/r/shuohang/seqmatchseq/): docker pull shuohang/seqmatchseq:1.0

After installation, just run the following codes:
```
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt      shuohang/seqmatchseq:1.0 /bin/bash -c "sh preprocess.sh squad"
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt/main shuohang/seqmatchseq:1.0 /bin/bash -c "th mainDt.lua"
```

# Sequences Maching in Word Level
### Requirements
- [Torch7](https://github.com/torch/torch7)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- Python 2.7

### Datasets
- [The Stanford Natural Language Inference (SNLI) Corpus](http://nlp.stanford.edu/projects/snli/)
- [MovieQA: Story Understanding Benchmark](http://movieqa.cs.toronto.edu/home/)
- [InsuranceQA Corpus V1: Answer Selection Task](https://github.com/shuzi/insuranceQA)
- [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/data/glove.840B.300d.zip)

For now, this code only support SNLI and WikiQA data sets.

### Usage
SNLI task:
```
sh preprocess.sh snli
cd main
th main.lua -task snli -model compAggSNLI -comp_type submul -learning_rate 0.002 -mem_dim 150 -dropoutP 0.3 
```
WikiQA task:
```
sh preprocess.sh wikiqa (Please first dowload the file "WikiQACorpus.zip" to the path data/wikiqa/ through address: https://www.microsoft.com/en-us/download/details.aspx?id=52419)
cd main
th main.lua -task wikiqa -model compAggWikiqa -comp_type mul -learning_rate 0.004 -dropoutP 0.04 -batch_size 10 -mem_dim 150 
```

- model (model name) : compAggSNLI for "SNLI" / compAggWikiqa for "WikiQA"
- comp_type (8 different types of word comparison): submul / sub / mul / weightsub / weightmul / bilinear / concate / cos

### Docker
You may try to use Docker for running the code.
- [Docker Install](https://github.com/codalab/codalab-worksheets/wiki/Installing-Docker)
- [Image](https://hub.docker.com/r/shuohang/seqmatchseq/): docker pull shuohang/seqmatchseq:1.0

After installation, just run the following codes:
For SNLI:
```
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt      shuohang/transition:1.1 /bin/bash -c "sh preprocess.sh snli"
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt/main shuohang/transition:1.1 /bin/bash -c "th main.lua -task snli -model compAggSNLI -comp_type submul -learning_rate 0.002 -mem_dim 150 -dropoutP 0.3"
```
For WikiQA
```
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt      shuohang/seqmatchseq:1.0 /bin/bash -c "sh preprocess.sh wikiqa"
docker run -it -v /PATH/SeqMatchSeq:/opt --rm -w /opt/main shuohang/seqmatchseq:1.0 /bin/bash -c "th main.lua -task wikiqa -model compAggWikiqa -comp_type mul -learning_rate 0.004 -dropoutP 0.04 -batch_size 10 -mem_dim 150"
```
# Copyright
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
