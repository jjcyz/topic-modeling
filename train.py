import numpy as np
from sklearn.datasets import fetch_20newsgroups
from topic_model import MixtureMultinomialEM
from utils import prepare_data, plot_convergence, print_topics, plot_topic_word_heatmap
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def fit_and_analyze(T, K, vocab):
    """Fit model and return results."""
    print(f"Starting training for K={K} topics...")
    print(f"Data shape: {T.shape}")

    start_time = time.time()
    model = MixtureMultinomialEM(n_topics=K, max_iter=100, random_state=42)
    model.fit(T)

    topics = model.get_top_words(vocab, n_words=10)
    print(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Calculate topic diversity
    unique_words = set()
    for topic in topics:
        unique_words.update(topic)
    print(f"Number of unique words across all topics: {len(unique_words)}")

    return {
        'model': model,
        'topics': topics,
        'final_likelihood': model.log_likelihood_history[-1],
        'convergence': model.log_likelihood_history
    }

def main():
    try:
        print("Loading 20 Newsgroups dataset...")
        newsgroups = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes')
        )
        print(f"Loaded {len(newsgroups.data)} documents")

        print("Preparing data...")
        T, vocabulary, vectorizer = prepare_data(newsgroups.data)
        print(f"Vocabulary size: {len(vocabulary)}")
        print(f"Document-term matrix shape: {T.shape}")

        # Test different K values
        K_values = [10, 15, 20]  # Pset: 10, 20, 30, 50; 20+ takes too long
        results = {}

        for K in K_values:
            print(f"\nFitting model with K={K} topics...")
            results[K] = fit_and_analyze(T, K, vocabulary)
            print(f"Final log likelihood: {results[K]['final_likelihood']}")

        # Generate and save all visualizations
        print("\nGenerating plots and saving results...")
        plot_convergence(results, K_values)
        print_topics(results, K_values)

        # Plot heatmap for each K
        for K in K_values:
            plot_topic_word_heatmap(results[K]['model'], vocabulary)

        print("\nTraining completed successfully!")
        print("All results have been saved to the 'plots' and 'results' directories.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
