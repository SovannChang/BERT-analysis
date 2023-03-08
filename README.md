# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
### Original Authors Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova
#### Analysis by Sovann Chang

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
    
### Pre-Training

