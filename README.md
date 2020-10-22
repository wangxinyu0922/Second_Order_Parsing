# Second-Order Syntactic/Semantic Dependency Parser

An implementation of our AACL 2020 paper "Second-Order Neural Dependency Parsing with Message Passing and End-to-End Training" and a new version of our ACL 2019 paper "Second-Order Semantic Dependency Parsing with End-to-End Neural Networks".

The code is based on the old version of [`SuPar`](https://github.com/yzhangcs/parser)
<!-- Details and [hyperparameter choices](#Hyperparameters) are almost identical to those described in the paper, 
except that we provide the Eisner rather than MST algorithm to ensure well-formedness. 
Practically, projective decoding like Eisner is the best choice since PTB contains mostly (99.9%) projective trees.
 -->

Comparing with original code, we use MST instead of Eisner for syntactic dependency parsing. Our code is also able to concatenate word, POS tags, char and BERT embeddings as token representations.

## Requirements

* `python`: 3.7.0
* [`pytorch`](https://github.com/pytorch/pytorch): 1.3.0
* [`transformers`](https://github.com/huggingface/transformers): 2.1.1

## Datasets

The model is evaluated on the Stanford Dependency conversion ([v3.3.0](https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip)) of the English Penn Treebank with POS tags predicted by [Stanford POS tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip).

For all datasets, we follow the conventional data splits:

* Train: 02-21 (39,832 sentences)
* Dev: 22 (1,700 sentences)
* Test: 23 (2,416 sentences)

## Performance for Syntactic Dependency Parsing

| MODEL          |  UAS  |  LAS  | Speed (Sents/s) |
| ------------- | :---: | :---: | :-------------: |
| Single1O + TAG + MST     | 95.75 | 94.04 |   1123  |
| Local1O + TAG + MST       | 95.83 | 94.23 | 1150 | 
| Single2O + TAG + MST      | 95.86 | 94.19 | 966 |
| Local2O + TAG + MST       | 95.98 | 94.34 | 1006 |
| Local2O + MST (Best)      | 96.12 | 94.47 | 1006 |
| CRF2O (Best) [(Zhang et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.302/)| 96.14 | 94.49 | 400 |
<!-- | CHAR          | 95.99 | 94.38 |     1464.59     |
| CHAR + Eisner | 96.02 | 94.41 |     323.73      |
| BERT          | 96.64 | 95.11 |     438.72      |
| BERT + Eisner | 96.65 | 95.12 |     214.68      | -->

Where `1O` represents first-order, `2O` reperesents second-order, `Single` represents binary classification `Local` represents head-selection. The results are averaged over 5 times, `Best` represents the single test results based on best development performance. Punctuation is ignored in all evaluation metrics for PTB. 


## Usage

You can start the training, evaluation and prediction process by using subcommands registered in `parser.cmds`.

To train a syntactic parser, run:

```sh
$ CUDA_VISIBLE_DEVICES=0 python3 -u run.py train  --conf config/3iter_100binary_0init_ptb_full_tree_0.cfg
```

To train a semantic parser, you can modify the dataset split in the config file. Then set `tree = False` and `binary = True`. Moreover, based on the binary structure, you can train a [Enhanced Universal Dependencies](https://universaldependencies.org/iwpt20/data.html) (EUD) parser as well. But for better training a EUD parser, please use [MultilangStructureKD](https://github.com/Alibaba-NLP/MultilangStructureKD).

All the data files must follow the [CoNLL-U format](https://universaldependencies.org/format.html). 

## Other codes
* Tensorflow version of semantic dependency parser: [Second_Order_SDP](https://github.com/wangxinyu0922/Second_Order_SDP).
* Pytorch version of enhanced universal dependencies parser: [MultilangStructureKD](https://github.com/Alibaba-NLP/MultilangStructureKD).
* An application for Mean-Field Variational Inference to Sequence Labeling: [AIN](https://github.com/Alibaba-NLP/AIN).
* The PyTorch Version of Biaffine Parser: [parser](https://github.com/yzhangcs/parser).

## References

* [Second-Order Neural Dependency Parsing with Message Passing and End-to-End Training](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/aacl20.pdf)
* [Second-Order Semantic Dependency Parsing with End-to-End Neural Networks](https://www.aclweb.org/anthology/P19-1454/)

