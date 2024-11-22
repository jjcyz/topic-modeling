import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from topic_model import MixtureMultinomialEM
from visualize_results import (plot_topic_coherence, plot_topic_similarity_heatmap,
                             plot_topic_evolution, plot_topic_distinctiveness,
                             save_topics_to_file)
import time

def prepare_data_improved(texts):
    """Improved text preprocessing."""
    # Extended stop words
    extra_stop_words = ['would', 'could', 'should', 'like', 'know', 'think', 'just',
                       'say', 'said', 'time', 'new', 'use', 'used', 'using']

    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        max_df=0.7,  # Remove words that appear in >70% of documents
        min_df=10,   # Remove words that appear in <10 documents
        token_pattern=r'(?u)\b[A-Za-z]{3,}\b',  # Only words with 3+ letters
        binary=False  # Keep count information
    )

    # Fit and transform the data
    T = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()

    # Remove single-letter words and common non-informative words
    mask = np.array([len(word) > 1 and word not in extra_stop_words
                    for word in vocabulary])
    T = T[:, mask]
    vocabulary = vocabulary[mask]

    return T, vocabulary

def analyze_model(K, T, vocabulary):
    """Train and analyze model for a specific K value."""
    print(f"\nAnalyzing model with K={K}")
    model = MixtureMultinomialEM(n_topics=K, max_iter=100, tol=1e-4, random_state=42)

    # Train model
    start_time = time.time()
    model.fit(T)
    training_time = time.time() - start_time

    # Get topics and coherence scores
    topics = model.get_top_words(vocabulary, n_words=10)
    coherence_scores = []

    print(f"\nTopics for K={K}:")
    for i, topic_words in enumerate(topics):
        probs = model.mu[i][np.argsort(-model.mu[i])[:10]]
        coherence = np.mean(probs)
        coherence_scores.append(coherence)
        print(f"\nTopic {i + 1}: {', '.join(topic_words)}")
        print(f"Coherence: {coherence:.4f}")

    # Calculate metrics
    avg_coherence = np.mean(coherence_scores)
    duplicate_topics = sum(1 for score in coherence_scores if score < 0.0001)

    # Generate visualizations
    plot_topic_coherence(topics, coherence_scores)
    plot_topic_similarity_heatmap(model)
    plot_topic_evolution(model.log_likelihood_history)
    plot_topic_distinctiveness(model)

    # Save results
    final_likelihood = model.log_likelihood_history[-1]
    save_topics_to_file(topics, coherence_scores, model, final_likelihood)

    return {
        'K': K,
        'avg_coherence': avg_coherence,
        'duplicate_topics': duplicate_topics,
        'training_time': training_time,
        'iterations': len(model.log_likelihood_history)-1,  # Subtract 1 for iteration count
        'final_likelihood': final_likelihood
    }

def main():
    # Load and prepare data
    print("Loading data...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    print("Preparing data...")
    T, vocabulary = prepare_data_improved(newsgroups.data)
    print(f"Processed data shape: {T.shape}")

    # Analyze different K values
    k_values = [10, 15, 20]
    results = []

    for K in k_values:
        result = analyze_model(K, T, vocabulary)
        results.append(result)

    # Compare results
    print("\nComparison of different K values:")
    print("==================================")
    for result in results:
        print(f"\nK = {result['K']}:")
        print(f"Average coherence: {result['avg_coherence']:.4f}")
        print(f"Duplicate topics: {result['duplicate_topics']}")
        print(f"Training time: {result['training_time']:.2f} seconds")
        print(f"Iterations to converge: {result['iterations']}")
        print(f"Final log likelihood: {result['final_likelihood']:.6e}")

if __name__ == "__main__":
    main()
