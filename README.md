# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
### Original Authors Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova
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

![Model Architectures](https://user-images.githubusercontent.com/59686399/223591492-69761df3-d0fc-4439-846f-a8df4fffb62b.png)

<br>

#  BERT
### How BERT Surpasses These Models
BERT incorporates a Masked Language Model (MLM) into its pre-training. The MLM randomly masks some of the input tokens and tries to predict the missing tokens from the surrounding context. This allows the model to use context from both sides of the target token when making predictions. BERT also uses Next Sentence Prediction (NSP) as a part of its pre-training.

<br>

### The Inner Workings of BERT
BERT's framework consists of two steps. The first is pre-training, done on unlabeled data over different tasks. In the second step, the BERT model is initialized with the pre-trained parameters, and all of the parameters are altered using labeled data from the downstream tasks. Each downstream task has separate models, even though they are initialized with the same pre-trained parameters.

### QUESTION: Is this second step a feature-based approach, or a fine-tuning approach?

<br>

### Architecture
BERT uses near-identical architecture for the pre-trained model and downstream models. 
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
  - Only the article text is used. not headers, lists, tables, etc.

It is importatnt to use document-level data instead of "shuffled sentence-level" data because it allows the model "to extract long, contiguous sequences."

###### MLM
The Masked Language Model randomly sets aside 15% of the input tokens. Because fine-tuning won't involve [MASK] tokens, the selected tokens are not all replaced with [MASK] tokens in order to reduce the mismatch between the tasks. Of the 15% set aside, 80% are replaced by [MASK], 10% are replaced by a random token, and 10% are unchanged. The advantage of this is that the transformer does not know which words have been replaced, so it cannot discard any token representations. After the tokens are replaced, the hidden vectors are used to predict the original tokens.

###### NSP
Next Sentence Prediction is pretty simple. The model is shown 2 sentences: Sentence A and Sentence B. Half of the time, Sentence B is the acatual sentence following Sentence A in the text, and half of the time, it is a random sentence. The model must decide which is the truth. The purpose of NSP is to capture the relationship between sentences.

<br>

### Fine-Tuning
The general strategy for fine-tuning BERT was to plug in task-specific inputs and outputs and fine-tune from end-to-end. BERT was fine-tuned using 4 datasets.

###### GLUE
The General Language Understanding Evaluation (GLUE) dataset is "a collection of diverse natural language understanding tasks." BERT's performance on GLUE is shown below.

![image](https://user-images.githubusercontent.com/59686399/223651737-e9b95895-014a-4e8c-92f1-62f4bd794206.png)

###### SQuAD 1.1
The Stanford Question Answering (SQuAD) dataset is contains 100,000 crowd-sourced question/answer pairs, with the questions asked about passages drawn from Wikipedia. The other models that BERT is being compared to were allowed to be trained on any data, so to even the playing field, BERT was first trained on TriviaQA, another 650,000 question/answer pairs. BERT's performance on SQuAD 1.1 is shown below.

![image](https://user-images.githubusercontent.com/59686399/223651809-398c47d9-051d-4e5f-ac8a-708b26096201.png)

###### SQuAD 2.0
SQuAD 2.0 adds the possibility that the answer to the question does not exist within the context in order to mimic reality. This time, BERT was not trained on TriviaQA first. BERT was also given a larger batch size and fewer epochs than when it was run on SQuAD 1.1. BERT's performance on SQuAD 2.0 is shown below.

![image](https://user-images.githubusercontent.com/59686399/223651898-c272fff1-8b78-44fa-972c-55e2ccbe94ed.png)


###### SWAG
The Situations With Adversarial Generations (SWAG) dataset is a collection of 113,000 examples of common-sense inference problems. Given a sentence, BERT should choose the most likely continuation. The input to BERT was four different sequences, each a concatenation of the prompt sentence and a possible continuation. BERT's performance on SWAG is shown below.

![image](https://user-images.githubusercontent.com/59686399/223651982-6893dc09-c804-4301-8368-af3b71a1684a.png)
