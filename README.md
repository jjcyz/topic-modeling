# Topic Modeling with EM Algorithm

Implementation of the Expectation-Maximization (EM) algorithm for topic modeling on the 20 Newsgroups dataset. This project explores topic modeling with different numbers of topics (K=10, 20, 30, 50) and includes visualizations of the results.

## Overview

The project implements topic modeling using a mixture of multinomials model. For each document:
1. A topic is chosen according to probability π_k
2. Words are generated according to that topic's multinomial distribution
3. The EM algorithm is used to find the maximum likelihood estimates of the parameters

## Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jjczy/topic_modeling.git
cd topic_modeling
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python improved_train.py
```

This will:
- Load the 20 Newsgroups dataset
- Preprocess the text data
- Train models with different K values (10, 20, 30, 50)
- Generate visualizations
- Save results in plots/ and results/ directories

## Files

### Core Implementation
- `topic_model.py`: Implementation of EM algorithm for mixture of multinomials
- `improved_train.py`: Training script with visualization and analysis functions

### Generated Results
- `plots/`: Directory containing visualization plots
  - `topic_coherence_k{K}.png`: Topic coherence scores
  - `topic_similarity_k{K}.png`: Topic similarity heatmaps
  - `convergence_evolution_k{K}.png`: Convergence plots
  - `topic_distinctiveness_k{K}.png`: t-SNE visualizations
  - `k_comparison.png`: Comparison of different K values

- `results/`: Directory containing analysis results
  - `topics_k{K}.txt`: Detailed topic analysis for each K value

## Visualizations

The project generates several types of visualizations:

1. **Topic Coherence Plots**
   - Bar plots showing coherence scores for each topic
   - Higher scores indicate more coherent topics

2. **Topic Similarity Heatmaps**
   - Visualize similarity between topics
   - Darker colors indicate higher similarity

3. **Convergence Plots**
   - Show log likelihood evolution during training
   - Helps assess convergence behavior

4. **Topic Distinctiveness**
   - t-SNE visualization of topic relationships
   - Spatially separated topics are more distinct

## Implementation Details

### Preprocessing
- Removes common stop words and domain-specific common words
- Filters words based on document frequency
- Uses only words with 4+ characters

### Model Parameters
- Number of topics (K): 10, 20, 30, 50
- Maximum iterations: 100
- Convergence tolerance: 1e-4
- Vocabulary size: 400 words

### Evaluation Metrics
- Topic coherence scores
- Topic similarity measures
- Convergence analysis
- Training time comparisons

## Results

The analysis shows:
1. Topic quality varies with different K values
2. Some topics appear redundant at higher K values
3. Convergence behavior depends on K
4. Training time increases with K

## Acknowledgments

- Based on the EM algorithm for mixture models
- Uses the 20 Newsgroups dataset from scikit-learn
- Implements visualization techniques from various scientific computing libraries

## Author

Jessica Zhou

## References

1. The 20 Newsgroups Dataset
2. EM Algorithm for Mixture Models
3. scikit-learn Documentation
4. scipy and numpy Documentation
