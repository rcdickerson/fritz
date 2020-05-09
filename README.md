FRITZ: Faulty Reasoning over Input TransformationZ
==================================================

Code for the evaluation of an input dataset transformation based approach for revealing bad heuristics in natural language inference models. This was part of a final project for Dan Goldwasser's Natural Language Processing (CS577) course at Purdue University.

Background
----------

Textual Entailment is a relation that holds between two sentences P (the premise) and H (the hypothesis) when H logically follows from P. For example, "The lawyer drove his car to visit a client" entails "The lawyer drove a car," but not "The client drove a car." Natural Language Inference (NLI, also referred to as Recognizing Textual Entailment or RTE) is a subfield of natural language processing that attempts to model and classify textual entailment. In a 2019 paper entitled [Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference](https://arxiv.org/pdf/1902.01007.pdf), McCoy et al. posit the existence of unsound heuristics that may be inadvertantly trained into NLI models. The authors develop a dataset called HANS designed to perform poorly on models exhibiting three different unsound NLI heuristics and demonstrate the presence of these heuristics in four previously developed NLI models.

The goal of this project is to experiment with exposing the presence of similar heuristics by making transformations to the datasets used in training of the NLI model. These transformations consist of things like swapping or shuffling words, replacing words with synonyms or antonyms, and adding or removing words from sentences. The code in this repository was used to evaluate the performance of two different NLI models on various dataset transformations.

Models
------
TODO

Transformations
---------------
TODO

Evaluation Datasets
-------------------
TODO
