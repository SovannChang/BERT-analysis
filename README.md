# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
### Original authors Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova
#### Analysis by Sovann Chang

Link to the paper: https://arxiv.org/abs/1810.04805

# Background
### Conceptual Background
Feature-based approach vs. Fine-tuning approach
  - In a feature-based approach, model A is trained, and its representations are used as an input to model B, which is downstream from model A.
  - In a fine-tuning approach, model A is trained, and model B (downstream from A) is refined by tweaking model A's parameters, adjusting the entire system.

### Transformers Before BERT
Before BERT, there were two types of transformers:

  - Models like OpenAI GPT were powerful, but only used unidirectional attention, meaning that the target token could only get context from tokens preceding OR following it - not both.
  - Models like ELMo separately trained forward (left to right) and backward (right to left) unidirectional language models, concatenating their representations of the target token after training. This allowed the target token to attend to tokens before and after it, but was twice as expensive as a single model, and only provided a shallow representation of the token.

<p align="center">
  <img src="https://user-images.githubusercontent.com/59686399/223591492-69761df3-d0fc-4439-846f-a8df4fffb62b.png" />
<p>

<br>

#  BERT
### How BERT Surpasses These Models
BERT incorporates a Masked Language Model (MLM) into its pre-training. The MLM randomly masks some of the input tokens and tries to predict the missing tokens from the surrounding context. This allows the model to use context from both sides of the target token when making predictions. BERT also uses Next Sentence Prediction (NSP) as a part of its pre-training.

<p align="center">
  <img src="https://user-images.githubusercontent.com/59686399/223962799-9e6616db-4606-4d12-a552-1bf59a38b162.png" width="300" />
</p>


<br>

### The Inner Workings of BERT
BERT's framework consists of two steps. The first is pre-training, done on unlabeled data over different tasks. In the second step, the BERT model is initialized with the pre-trained parameters, and all of the parameters are altered using labeled data from the downstream tasks. Each downstream task has separate models, even though they are initialized with the same pre-trained parameters.

<br>

### QUESTION: Is this second step a feature-based approach, or a fine-tuning approach?

<br>

### Architecture
BERT uses near-identical architecture for the pre-trained model and downstream models. 

![image](https://user-images.githubusercontent.com/59686399/223914863-82ab937d-f926-4d35-9e3f-4d4c4cbcc2c5.png)
The BERT architecture (credit towardsdatascience)

<br>

BERT has two main model sizes: 
  - BERT base 
    - 12 transformer layers, 768-dimensional hidden vector, 12 attention heads
    - 110 million parameters
  - BERT large 
    - 24 transformer layers, 1024-dimensional hidden vector, 16 attention heads 
    - 340 million parameters

### Inputs
The entire input is represented as one sequence. Different sentences (e.g. question/answer) are separated using the [SEP] token. Learned embeddings that represent which sentence the token is in are added to the tokens.
   
<br>
   
### Pre-Training
As stated above, BERT is trained on a Masked Language Model and Next Sentence Prediction

##### Data
There are two primary datasets used in pre-training:
- BooksCorpus contains around 800 million words
- English Wikipedia contains around 2.5 billion words
  - Only the article text is used, not headers, lists, tables, etc.

It is importatnt to use document-level data instead of "shuffled sentence-level" data because it allows the model "to extract long, contiguous sequences."

###### MLM
The Masked Language Model randomly sets aside 15% of the input tokens. Because fine-tuning won't involve [MASK] tokens, the selected tokens are not all replaced with [MASK] tokens in order to reduce the mismatch between the tasks. Of the 15% set aside, 80% are replaced by [MASK], 10% are replaced by a random token, and 10% are unchanged. The advantage of this is that the transformer does not know which words have been replaced, so it cannot discard any token representations. After the tokens are replaced, the hidden vectors are used to predict the original tokens.

###### NSP
Next Sentence Prediction is pretty simple. The model is shown 2 sentences: Sentence A and Sentence B. Half of the time, Sentence B is the actual sentence following Sentence A in the text, and half of the time, it is a random sentence. The model must decide which is the truth. The purpose of NSP is to capture the relationship between sentences.

<br>

### Fine-Tuning
The general strategy for fine-tuning BERT was to plug in task-specific inputs and outputs and fine-tune from end-to-end. BERT was fine-tuned using 4 datasets.

###### GLUE
The General Language Understanding Evaluation (GLUE) dataset is "a collection of diverse natural language understanding tasks." BERT's performance on GLUE is shown below.

![image](https://user-images.githubusercontent.com/59686399/223651737-e9b95895-014a-4e8c-92f1-62f4bd794206.png)

###### SQuAD 1.1
The Stanford Question Answering (SQuAD) dataset is contains 100,000 crowd-sourced question/answer pairs, with the questions asked about passages drawn from Wikipedia. The other models that BERT is being compared to were allowed to be trained on any data, so to even the playing field, BERT was first trained on TriviaQA, another 650,000 question/answer pairs. BERT's performance on SQuAD 1.1 is shown below.

<img src="https://user-images.githubusercontent.com/59686399/223651809-398c47d9-051d-4e5f-ac8a-708b26096201.png" width="350" />

###### SQuAD 2.0
SQuAD 2.0 adds the possibility that the answer to the question does not exist within the context in order to mimic reality. This time, BERT was not trained on TriviaQA first. BERT was also given a larger batch size and fewer epochs than when it was run on SQuAD 1.1. BERT's performance on SQuAD 2.0 is shown below.

<img src="https://user-images.githubusercontent.com/59686399/223651898-c272fff1-8b78-44fa-972c-55e2ccbe94ed.png" width="350" />

###### SWAG
The Situations With Adversarial Generations (SWAG) dataset is a collection of 113,000 examples of common-sense inference problems. Given a sentence, BERT should choose the most likely continuation. The input to BERT was four different sequences, each a concatenation of the prompt sentence and a possible continuation. BERT's performance on SWAG is shown below.

<img src="https://user-images.githubusercontent.com/59686399/223651982-6893dc09-c804-4301-8368-af3b71a1684a.png" width="250" />

<br>

### Ablation
Broadly speaking, ablation is testing to determine the impact of an aspect by removing it and observing the changes. The authors investigate the effects of altering three aspects of BERT:

#### Effect of Pre-Training Tasks
Using the same data, fine-tuning scheme, and hyperparameters, base BERT was compared to a version with an MLM but no NSP training, and to another version that has no NSP training and replaces the MLM with a Left-To-Right (LTR) language model. The results are described below, along with a table.
  - Removing NSP hurt performance significantly on 3 of the 5 tasks
  - After removing NSP, swapping MLM for LTR hurt performance extremely on 2 of the 5 tasks and significantly on one
  - For SQuAD, the authors attempted to give it a fair shot by adding a randomly initialized bidirectional LSTM
    - Improved performance heavily, but still significantly worse than pre-trained bidirectional model

<img src="https://user-images.githubusercontent.com/59686399/223673753-884d4ee1-916b-4ad8-9b7f-350337b5ad09.png" width="350" />

#### Effect of Model Size
To test the effects of changing the model's size, BERT was run 6 times with different hyperparameters. The tested hyperparameters were the number of transformer layers (L), the dimension of the hidden vector (H), and the number of attention heads (A). The results are shown in the table below. BERT base is the fourth row, and BERT large is the last row.

<img src="https://user-images.githubusercontent.com/59686399/223778044-b2b28efa-ebfb-418d-9927-cd131bc1ba3f.png" width="350" />

This analysis showed that larger models led to better performance on large-scale tasks, which was already known when this was published. However, it also showed that if a model has been sufficiently pre-trained, very small-scale tasks can also see significant improvements when the model is scaled up.

#### Feature-based Approach
The authors experimented with switching BERT to use a feature-based approach, where the pre-trained model produces features that remain unchanged when being passed to the downstream model. 

<br>

### QUESTION: What is a potential advantage of a feature-based approach?

<br><br>

One advantage of this approach is that some tasks require task-specific architecture, and can't be easily represented by a transformer encoder architecture. Feature-based learning is also more efficient than fine-tuning, since the representations of the training data are pre-computed only once and other, cheaper models run on those.

To test the feature-based BERT, the authors used the CoNLL-2003 Named Entity Recognition (NER) dataset. CoNLL-2003 contains around 20,000 examples of sentences containing named entities, and the BERT's job was to identify which words refer to named entities. The results of the testing are shown in the table below.

<img src="https://user-images.githubusercontent.com/59686399/223785939-55a2e4cc-30f8-415c-8b0e-be3684e615a6.png" width="350" />

The table shows that the best feature-based approach performs nearly as well as the fine-tuning approaches, which means that BERT is as effective when using feature-based learning as when it uses fine-tuning.

# Conclusion
Before BERT, rich, unsupervised pre-training had been shown to improve low-resource tasks using unidirectional models. BERT took this concept and applied it to a more powerful, bidirectional model. It showed that pre-trained representations could outperform task-specific architectures for both sentence-level and token-level tasks. BERT surpassed all other models on most tasks, and still performed well when its fine-tuning was replaced with feature-based learning.

#### Criticism
The authors chose not to detail the architecture of BERT, instead referencing a separate paper (Attention is All You Need) and suggesting the reader look there instead, since their implementation is "almost identical to the original." Personally, I think that it would have been useful for them to include at least a high-level overview and a diagram so that interested readers could get a general idea of how it worked rather than being stuck with the options of "read another paper" or "accept that you won't know how it's structured." 

As far as I can tell, the paper was very widely acepted. I could not find any other authors who disputed any of the paper. It has been cited over 60,000 times, and BERT has been used as the basis for many newer models.

<br>

## Links To Further Resources
  - BERT repository - https://github.com/google-research/bert
  - BERT Huggingface page - https://huggingface.co/bert-base-uncased
  - Transformer implementation and architecture - https://arxiv.org/abs/1706.03762
  - ELMo model - https://arxiv.org/abs/1802.05365
  - Generative pretraining on OpenAI GPT - https://paperswithcode.com/paper/improving-language-understanding-by
  - RoBERTa paper - https://arxiv.org/abs/1907.11692
    - RoBERTa may be the best follow-up to BERT so far, expanding its training data and adjusting its hyperparameters to achieve even better results
