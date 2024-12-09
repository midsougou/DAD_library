# DAD_library
Library for discreet sequences anomaly detection.


#### MarkovianTechniques

The `MarkovianTechniques` package implements various anomaly detection methods for discrete sequences based on the techniques discussed in the paper **"Anomaly Detection for Discrete Sequences: A Survey"**. These methods cover both fixed and advanced Markovian models, enabling flexible and powerful sequence modeling for anomaly detection tasks.

## Implemented Techniques

The package includes the following implementations, each corresponding to a specific technique from the paper:

- **FixedMarkovianBased.py**: Implements fixed-order Markovian techniques, where the next symbol is predicted using a fixed context size \(k-1\).
- **VariableMarkovianBased.py**: Implements variable-order Markovian techniques using probabilistic suffix trees (PSTs), allowing the context size to adapt dynamically.
- **SparseMarkovTransducer.py**: Implements sparse Markovian techniques using wildcards in the context, modeled as a sparse suffix tree for efficient storage and backoff.
- **SparseMarkovRIPPER.py**: Implements rule-based sparse Markovian techniques using a decision-tree approximation of the RIPPER algorithm to learn symbolic rules for context-symbol relationships.

## Supporting Modules

- **SuffixTreeNode.py**: Contains the data structures and logic for managing suffix trees, which are a core component of variable and sparse Markovian techniques.

## Usage

The notebook gives a first implementation on how to use the different techniques in a simulated data environment.
