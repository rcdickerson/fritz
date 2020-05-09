FRITZ: Faulty Reasoning over Input TransformationZ
==================================================

Code for the evaluation of an input dataset transformation based approach for revealing bad heuristics in natural language inference models. This was part of a final project for Dan Goldwasser's Natural Language Processing (CS577) course at Purdue University.


Background
----------

Textual Entailment is a relation that holds between two sentences P (the premise) and H (the hypothesis) when H logically follows from P. For example, "The lawyer drove his car to visit a client" entails "The lawyer drove a car," but not "The client drove a car." Natural Language Inference (NLI, also referred to as Recognizing Textual Entailment or RTE) is a subfield of natural language processing that attempts to model and classify textual entailment. In a 2019 paper entitled [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://arxiv.org/pdf/1902.01007.pdf), McCoy et al. posit the existence of unsound heuristics that may be inadvertantly trained into NLI models. The authors develop a dataset called HANS designed to perform poorly on models exhibiting three different unsound NLI heuristics and demonstrate the presence of these heuristics in four previously developed NLI models.

The goal of this project is to experiment with exposing the presence of similar heuristics by making transformations to the datasets used in training of the NLI models themselves. These transformations consist of things like swapping or shuffling words, replacing words with synonyms or antonyms, and adding or removing words from sentences. The idea is related to adversarial input perterbations for neural classifiers (c.f. [Moosavi-Dezfooli et al.](https://arxiv.org/abs/1610.08401)), data noising for neural network smoothing (c.f. [Xie et al.](https://arxiv.org/abs/1703.02573)), and specific data transformations were inspired by [Wei and Zou](https://arxiv.org/abs/1901.11196). 

The code in this repository was used to evaluate the performance of two different NLI models on various dataset transformations.


Setup
-----

Code was written in Python 3, and library dependencies are listed in `requirements.txt`. Setting up with [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) in Linux would look like:

```bash
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Models
------

Evaluation was carried out on two NLI models:

* The Decomposable Attention (DA) model as presented by [Parikh et al.](https://www.aclweb.org/anthology/D16-1244/)
* The Enhanced Sequential Inference Model (ESIM) as presented by [Chen et al.](https://www.aclweb.org/anthology/P17-1152/)

Models were trained on the [MNLI corpus](https://cims.nyu.edu/~sbowman/multinli/) using the stock implementations found in the [AllenNLP library](https://allennlp.org/). 

The model configurations used in these experiments are located in the `model_params` directory. Training was done via the AllenNLP CLI:

```bash
allennlp train -s ./da model_params/da.jsonnet 
allennlp train -s ./esim model_params/esim.jsonnet 
```
MNLI training data is assumed to be located at `../multinli_1.0` and the batch size is set to 32, but these (and other) parameters may be changed by editing the appropriate `jsonnet` files in `model_params`.


Transformations
---------------
TODO

Evaluation Datasets
-------------------
TODO
