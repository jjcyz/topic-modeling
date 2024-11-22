import numpy as np
from sklearn.datasets import fetch_20newsgroups
from topic_model import MixtureMultinomialEM
from utils import prepare_data, plot_convergence, print_topics, plot_topic_word_heatmap
import time

def main():
    # Load a subset of the data
    print("Loading subset of 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset='train',  # Only use training set (about half the data)
        remove=('headers', 'footers', 'quotes')
    )
    print(f"Loaded {len(newsgroups.data)} documents")

    # Prepare data with smaller vocabulary
    print("Preparing data...")
    T, vocabulary, vectorizer = prepare_data(
        newsgroups.data,
        max_features=500,  # Reduced vocabulary size
        max_df=0.95,
        min_df=10
    )
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Document-term matrix shape: {T.shape}")

    # Test different K values
    K_values = [10, 20]
    results = {}

    for K in K_values:
        print(f"\nFitting model with K={K} topics...")
        start_time = time.time()

        # Initialize model with faster convergence parameters
        model = MixtureMultinomialEM(
            n_topics=K,
            max_iter=50,  # Reduced maximum iterations
            tol=1e-3,     # Increased tolerance for faster convergence
            random_state=42
        )

        model.fit(T)
        topics = model.get_top_words(vocabulary, n_words=10)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        results[K] = {
            'model': model,
            'topics': topics,
            'final_likelihood': model.log_likelihood_history[-1],
            'convergence': model.log_likelihood_history
        }

    # Plot results
    plot_convergence(results, K_values)
    print_topics(results, K_values)
    for K in K_values:
        plot_topic_word_heatmap(results[K]['model'], vocabulary)

if __name__ == "__main__":
    main()
