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

Code was written in Python 3, and library dependencies are listed in `requirements.txt`. Setting up with [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) in Linux might look like:

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
allennlp train -s ./models/da ./model_params/da.jsonnet 
allennlp train -s ./models/esim ./model_params/esim.jsonnet 
```
Training data is assumed to be located at `../multinli_1.0` and the batch size is set to 32, but these (and other) parameters may be changed by editing the appropriate `jsonnet` file in `model_params`.


Transformations
---------------

All transformations are implemented in `perturb.py`:

```
usage: perturb.py [-h] [-o OUTPUT]
                  [-t {shuffle,shuffleprem,premsubseq,neghyp,negprem,hverbsyn,hverbant,hadvsyn,hadvant,hnounsyn,hnounant}]
                  infile

Perturb SNLI / MNLI style data

positional arguments:
  infile                file path of the input data to manipulate

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        file path to output perturbed data
  -t, --type {shuffle,shuffleprem,premsubseq,neghyp,negprem,hverbsyn,hverbant,hadvsyn,hadvant,hnounsyn,hnounant}
                        the type of perturbation to perform, default is shuffle
```

For example, transforming `eval_sets/mnli_all.tsv` using the `premsubseq` transformation would look like: 

```bash
python ./perturb.py ./eval_sets/mnli_all.tsv -t premsubseq
```

Available transforms are:

* **shuffle**: arbitrarily reorders the words in both the premise and hypothesis.
* **shuffleprem**: sets the hypothesis to an arbitrary reordering of the words in the premise.
* **premsubseq**: sets the hypothesis to an arbitrary subsequence of the premise.
* **neghyp**: negates the hypothesis by prepending it with the clause "It is not the case that".
* **negprem**: negates the premise by prepending it with the clause "It is not the case that".
* **hXsyn** (X = verb, adv, or noun): randomly replaces X part of speech in the hypothesis with a synonym using [WordNet](https://wordnet.princeton.edu/)
* **hXant** (X = verb, adv, or noun): randomly replaces X part of speech in the hypothesis with an antonym using [WordNet](https://wordnet.princeton.edu/)


Evaluation Datasets
-------------------

The `eval_sets` directory contains the MNLI development set, the HANS dataset, and `perturb.py`-generated transformations of the MNLI development set using each of the transform types listed above.

Accuracy of the models over these datasets may be evaluated using `eval_accuracy.py`:

```
usage: eval_accuracy.py [-h] --model_params MODEL_PARAMS --model_dir MODEL_DIR --eval_set EVAL_SET

Evaluate dataset accuracy on a variety of NLI models.

optional arguments:
  -h, --help            show this help message and exit
  --model_params MODEL_PARAMS
                        the .jsonnet model configuration
  --model_dir MODEL_DIR
                        the model serialization directory
  --eval_set EVAL_SET   the evaluation data set
```

For example, evaluating the accuracy of the DA model over `eval_sets/premsubseq.tsv` would look like:

```bash
python ./eval_accuracy.py --model_params model_params/da.jsonnet --model_dir ./models/da --eval_set ./eval_sets/premsubseq.tsv
```

assuming a model trained on `model_params/da.jsonnet` located at `models/da`.


Results
-------

Output from `eval_accuracy.py` on each of the datasets in `eval_sets` is listed in `results_da.txt` for runs using the DA model and `results_esim.txt` for runs using the ESIM model.
