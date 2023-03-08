# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
### Original Authors Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova
#### Analysis done by Sovann Chang

# Overview
### Term Background

### Transformers Before BERT
Before BERT, there were two types of transformers:

  - Models like OpenAI GPT were powerful, but only used unidirectional attention, meaning that the target token could only get context from tokens preceding OR following it - not both.
  - Models like ELMo separately trained forward (left to right) and backward (right to left) unidirectional language models, concatenating their representations of the target token after training. This allowed the target token to attend to tokens before and after it, but was twice as expensive as a single model, and only provided a shallow representation of the token.

![Model Architectures](https://user-images.githubusercontent.com/59686399/223591492-69761df3-d0fc-4439-846f-a8df4fffb62b.png)

### How BERT Surpasses These Models
BERT 
