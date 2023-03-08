# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
### Original Authors Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova
#### Analysis done by Sovann Chang

# Overview
### Transformers Before BERT
Before BERT, there were two types of transformers:

  - Models like OpenAI GPT were powerful, but only used unidirectional attention, meaning that the target token could only get context from tokens preceding OR following it - not both.
  - Models like ELMo separately trained forward (left to right) and backward (right to left) unidirectional language models, concatenating their representations of the target token after training. This allowed the target token to attend to tokens before and after it, but is twice as expensive as a single model, and only provides a shallow representation of the token.
