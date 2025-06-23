# Natural Language Processing Course Notes

This repository contains comprehensive notes and concepts from the Natural Language Processing course. The course covers fundamental topics in NLP, providing both theoretical foundations and practical applications using modern deep learning techniques.

## File Overview

### Module_001_01 - Word Vectors.ipynb
- Introduces the fundamental concepts of word vectors and embeddings
- Covers the evolution from sparse representations to dense word vectors
- Explains Word2Vec framework including Skip-Gram and CBOW architectures
- Details the objective function and softmax probability calculations
- Discusses the properties of word vectors and similarity measures

### Module_001_02 - Word Vectors and Language Models.ipynb  
- Explores optimisation basics including gradient descent and stochastic gradient descent
- Covers Word2Vec training algorithms and negative sampling techniques
- Explains Skip-Gram vs CBOW approaches with detailed examples
- Discusses evaluation methods for word vectors (intrinsic vs extrinsic)
- Introduces superposition concept in word vector representations
- Covers window-based classification and neural dependency parsing foundations

### Module_002 - Neural_NLP.ipynb
- **Part 1: Backpropagation and Neural Network Basics**
  - Reviews neural network fundamentals and matrix notation
  - Covers activation functions including sigmoid, tanh, ReLU, and modern variants
  - Explains gradient computation and the chain rule in detail
  - Demonstrates backpropagation with step-by-step examples
  - Introduces automatic differentiation concepts
- **Part 2: Dependency Parsing**
  - Covers syntactic structure and dependency grammar
  - Explains ambiguous parsing examples and treebanks
  - Details transition-based dependency parsing algorithms
  - Introduces neural dependency parsing using deep learning
  - Discusses modern approaches to syntactic analysis

### Module_003 - LMs_RNNs_LSTMs.ipynb
- **Part 1: Recurrent Neural Networks**
  - Reviews neural network improvements including regularisation, dropout, vectorisation, and optimisers
  - Introduces language modelling fundamentals and n-gram models
  - Covers neural language modelling with fixed-window models
  - Explains RNN architecture, advantages, and limitations
  - Discusses RNN training procedures and challenges
- **Part 2: Sequence-to-Sequence Models and Machine Translation**
  - Details language model evaluation using perplexity metrics
  - Explains vanishing and exploding gradient problems in RNNs
  - Introduces LSTM architecture with gating mechanisms (forget, input, output gates)
  - Covers how LSTMs solve vanishing gradient problems
  - Discusses machine translation challenges and neural machine translation (NMT)
  - Explains sequence-to-sequence models and BLEU evaluation metrics
- **Part 3: Attention Mechanisms**
  - Introduces attention concepts and motivation for seq2seq improvements
  - Details attention weight computation and context vector generation
  - Covers various attention variants: dot-product, multiplicative, reduced-rank, and additive attention
  - Explains scoring functions and attention mechanism applications

### Module_004 - Transformers.ipynb
- **Part 1: From Recurrence to Attention-based Models**
  - Identifies limitations of RNNs: linear interaction distance and lack of parallelisation
  - Introduces attention mechanisms for direct modelling of token relationships
  - Explains self-attention concepts and scaled dot-product attention formula
  - Addresses barriers to self-attention: sequence order, nonlinearities, and causal masking
  - Details positional encodings (sinusoidal and learned) for sequence order
  - Covers the complete self-attention building block architecture
- **Part 2: Transformer Architecture and Trade-offs**
  - Explains transformer decoder components: embeddings, positional encodings, masked multi-head attention
  - Details residual connections (Add & Norm) and feed-forward networks
  - Covers transformer encoder architecture with unmasked attention
  - Discusses model advantages: parallelisation, long-range dependencies, excellent NLP performance
  - Addresses trade-offs: quadratic complexity, computational requirements, and positional encoding limitations

## Key Concepts Covered

### Word Representations
- Dense vs sparse word representations
- Word2Vec (Skip-Gram and CBOW architectures)
- Negative sampling and training optimisation
- Contextual vs non-contextual embeddings
- Word vector evaluation methodologies

### Neural Networks for NLP
- Backpropagation algorithm and automatic differentiation
- Activation functions and their derivatives
- Gradient descent optimisation techniques
- Matrix operations and computational efficiency

### Syntactic Analysis
- Dependency parsing and grammatical structures
- Transition-based parsing algorithms
- Neural approaches to dependency parsing
- Ambiguity resolution in natural language

### Language Modelling and Sequential Processing
- N-gram models and their limitations
- Fixed-window neural language models
- Recurrent Neural Networks (RNNs) for sequence processing
- Language model evaluation using perplexity

### Advanced Sequential Architectures
- Vanishing and exploding gradient problems
- LSTM architecture and gating mechanisms
- Gradient flow improvements in deep networks
- Sequence-to-sequence models for translation

### Machine Translation
- Neural machine translation (NMT) fundamentals
- Encoder-decoder architectures
- BLEU evaluation metrics
- Translation quality assessment

### Attention Mechanisms
- Attention concept and motivation
- Attention weight computation and scoring functions
- Attention variants (dot-product, multiplicative, additive)
- Context vector generation and applications

### Transformer Architecture
- Self-attention mechanisms and scaled dot-product attention
- Multi-head attention and representation subspaces
- Positional encoding techniques (sinusoidal and learned)
- Encoder-decoder architectures and their differences
- Causal masking and autoregressive generation
- Residual connections and layer normalisation
- Feed-forward networks in transformer blocks
- Parallelisation advantages over recurrent models
- Computational complexity and scalability considerations

## Usage
These Jupyter notebooks serve as comprehensive study materials and reference guides for understanding modern NLP techniques. Each notebook contains detailed mathematical explanations, practical examples, and implementation concepts essential for NLP applications.

## Prerequisites
Basic understanding of:
- Linear algebra and calculus
- Probability and statistics
- Python programming
- Machine learning fundamentals

## License
These materials are shared with permission from Stanford University. Please use them responsibly and in accordance with the course's academic integrity policies.
