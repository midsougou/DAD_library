# DAD - Discreet (sequence) Anomaly Detection
Library for discreet sequences anomaly detection.

The `DAD` library implements various anomaly detection methods for discrete sequences based on the techniques discussed in the paper **"Anomaly Detection for Discrete Sequences: A Survey"**. So far the various anomaly detection techniques have been grouped in these three categories : 
- Kernel based techniques 
- Markovian techniques 
- window based techniques.
Because manipulating discreet sequences often implies sequencing continuous timeserie, this library contains a SAX algorithm to discreetize (or symbolize) any continuous timeserie into a discreet (or symbolic) sequence using the SAX alogirthm as shown in this paper **"Experiencing SAX: a Novel Symbolic Representation of Time Series"**.

### Kernel Based Techniques : 

These methods requires the speficiation of a Kernel to compute a distance, or similarity metric, between two sequences. From a training set, one compute the algorithm will be used to compute similarity matrix between all the sequences in the dataset (which can be quite heavy). This matrix will then be used to asses the anomaly score of any further test sample. The two classes implemented in the library are : 

- **KMedoids Based**: Performs Kmedoids on the similarity matrix and assesses test sample against the extracted medoids. 
- **Knearest Based**: Asses the test sample against all sequence of the training set and compute the k-th nearest sample. 

### Window Based Techniques : 

These methods parses the sequences into smaller chunks and assign anomaly score to each chunk. The overall anomly score of the sequence is then obtzained by an aggregate (mean, median, Kernel) of the anomaly score for all window. So far this library contains the following common window based techniques : 
- **Lookahead**: 
- **NormalDictionary**:
- **UnsupervisedSVM**: 

#### MarkovianTechniques

 These methods cover both fixed and advanced Markovian models, enabling flexible and powerful sequence modeling for anomaly detection tasks.

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
