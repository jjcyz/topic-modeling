import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import os

def prepare_data(texts, max_features=1000, max_df=0.95, min_df=10):  # Increased min_df
    """Prepare text data for topic modeling."""
    print("Vectorizing documents...")
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        max_df=max_df,
        min_df=min_df,
        token_pattern=r'(?u)\b[A-Za-z]+\b'  # Only keep words, no numbers or special chars
    )
    T = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()
    return T, vocabulary, vectorizer

def plot_convergence(results, k_values, save_dir='plots'):
    """Plot convergence of log likelihood for different K values."""
    # Create plots directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for K in k_values:
        plt.plot(results[K]['convergence'], label=f'K={K}')
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Convergence of EM Algorithm for Different K Values')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(os.path.join(save_dir, 'convergence_plot.png'))
    plt.close()
    print(f"Convergence plot saved to {save_dir}/convergence_plot.png")

def print_topics(results, k_values, save_dir='results'):
    """Print and save top words for each topic."""
    os.makedirs(save_dir, exist_ok=True)

    # Save to file
    with open(os.path.join(save_dir, 'topics.txt'), 'w') as f:
        for K in k_values:
            f.write(f"\nTop words for K={K}:\n")
            topics = results[K]['topics']

            # Calculate topic uniqueness
            word_sets = [set(topic) for topic in topics]
            unique_words = sum(len(set.difference(*[s for j, s in enumerate(word_sets) if j != i]))
                             for i in range(len(word_sets)))

            f.write(f"Uniqueness score: {unique_words/len(word_sets):.2f} unique words per topic\n\n")

            for i, topic_words in enumerate(topics):
                f.write(f"Topic {i + 1}: {', '.join(topic_words)}\n")

    print(f"Topics saved to {save_dir}/topics.txt")

def plot_topic_word_heatmap(model, vocabulary, n_top_words=20, save_dir='plots'):
    """Plot heatmap of top words in each topic."""
    os.makedirs(save_dir, exist_ok=True)

    # Get top words for each topic
    topics_matrix = []
    top_words = []

    for k in range(model.K):
        top_idx = np.argsort(-model.mu[k])[:n_top_words]
        topics_matrix.append(model.mu[k][top_idx])
        top_words.extend([vocabulary[i] for i in top_idx])

    # Remove duplicates while preserving order
    unique_words = list(dict.fromkeys(top_words))

    # Create heatmap data
    heatmap_data = np.zeros((model.K, len(unique_words)))
    for k in range(model.K):
        for i, word in enumerate(unique_words):
            idx = np.where(vocabulary == word)[0]
            if len(idx) > 0:
                heatmap_data[k, i] = model.mu[k, idx[0]]

    # Plot heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data,
                xticklabels=unique_words,
                yticklabels=[f'Topic {i+1}' for i in range(model.K)],
                cmap='YlOrRd')
    plt.xticks(rotation=45, ha='right')
    plt.title('Topic-Word Distribution Heatmap')
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(save_dir, 'topic_heatmap.png'))
    plt.close()
    print(f"Heatmap saved to {save_dir}/topic_heatmap.png")
