import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from topic_model import MixtureMultinomialEM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time

def prepare_data_improved(texts):
    """Improved text preprocessing with expanded stop words."""
    extra_stop_words = [
        # Common English words
        'would', 'could', 'should', 'like', 'know', 'think', 'just',
        'say', 'said', 'time', 'new', 'use', 'used', 'using',
        # Common fillers
        'test', 'hello', 'thanks', 'good', 'way', 'want', 'need',
        'tell', 'look', 'see', 'make', 'well', 'get', 'got', 'go',
        'going', 'went', 'come', 'came', 'take', 'took', 'think',
        # More specific stop words
        'actually', 'probably', 'maybe', 'seems', 'look', 'looking',
        'looks', 'seem', 'seemed', 'seems', 'thank', 'thanks', 'hello',
        'hi', 'hey', 'please', 'sorry', 'regards', 'regard', 'heard',
        'stuff', 'easy', 'wanted', 'simple', 'david', 'contact',
        'jewish', 'especially', 'difference'
    ]

    vectorizer = CountVectorizer(
        max_features=400,
        stop_words='english',
        max_df=0.5,
        min_df=20,
        token_pattern=r'(?u)\b[A-Za-z]{4,}\b',
        binary=False
    )

    T = vectorizer.fit_transform(texts).toarray()
    vocabulary = vectorizer.get_feature_names_out()

    mask = np.array([len(word) > 1 and word not in extra_stop_words
                    for word in vocabulary])
    T = T[:, mask]
    vocabulary = vocabulary[mask]

    return T, vocabulary

def create_visualizations(model, topics, coherence_scores, k_value):
    """Create all visualizations for a given model."""
    plot_topic_coherence(topics, coherence_scores, k_value)
    plot_topic_similarity_heatmap(model, k_value)
    plot_topic_evolution(model.log_likelihood_history, k_value)
    plot_topic_distinctiveness(model, k_value)

def plot_topic_coherence(topics, coherence_scores, k_value):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(1, len(coherence_scores) + 1), coherence_scores)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.xlabel('Topic Number')
    plt.ylabel('Coherence Score')
    plt.title(f'Topic Coherence Scores (K={k_value})')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'plots/topic_coherence_k{k_value}.png')
    plt.close()

def plot_topic_similarity_heatmap(model, k_value):
    similarity_matrix = np.zeros((model.K, model.K))
    for i in range(model.K):
        for j in range(model.K):
            similarity_matrix[i,j] = np.dot(model.mu[i], model.mu[j]) / \
                                   (np.linalg.norm(model.mu[i]) * np.linalg.norm(model.mu[j]))

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                xticklabels=[f'Topic {i+1}' for i in range(model.K)],
                yticklabels=[f'Topic {i+1}' for i in range(model.K)])
    plt.title(f'Topic Similarity Matrix (K={k_value})')
    plt.tight_layout()
    plt.savefig(f'plots/topic_similarity_k{k_value}.png')
    plt.close()

def plot_topic_evolution(log_likelihood_history, k_value):
    plt.figure(figsize=(10, 6))
    plt.plot(log_likelihood_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title(f'Model Convergence Over Iterations (K={k_value})')

    convergence_point = len(log_likelihood_history) - 1
    final_likelihood = log_likelihood_history[-1]
    plt.plot(convergence_point, final_likelihood, 'ro', markersize=10,
             label=f'Converged at iteration {convergence_point}')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'plots/convergence_evolution_k{k_value}.png')
    plt.close()

def plot_topic_distinctiveness(model, k_value):
    if model.K < 3:
        return

    perplexity = min(model.K - 1, 5)

    try:
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    n_iter=1000,
                    random_state=42)
        topic_coords = tsne.fit_transform(model.mu)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(topic_coords[:, 0], topic_coords[:, 1],
                            c=range(model.K), cmap='tab20', s=100)

        for i, (x, y) in enumerate(topic_coords):
            plt.annotate(f'Topic {i+1}', (x, y), xytext=(5, 5),
                        textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.7))

        plt.title(f'Topic Distinctiveness Visualization (K={k_value})')
        plt.colorbar(scatter, label='Topic Number')
        plt.savefig(f'plots/topic_distinctiveness_k{k_value}.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create topic distinctiveness plot: {str(e)}")

def plot_k_comparison(results):
    plt.figure(figsize=(12, 6))
    for k, data in results.items():
        plt.plot(data['log_likelihood_history'], label=f'K={k}')

    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Convergence of EM Algorithm for Different K Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/k_comparison.png')
    plt.close()

def analyze_model(model, vocabulary):
    """Analyze model results and compute coherence scores."""
    topics = model.get_top_words(vocabulary, n_words=10)
    coherence_scores = []

    for i, topic_words in enumerate(topics):
        probs = model.mu[i][np.argsort(-model.mu[i])[:10]]
        coherence = np.mean(probs)
        coherence_scores.append(coherence)

        print(f"\nTopic {i + 1}: {', '.join(topic_words)}")
        print(f"Topic coherence: {coherence:.4f}")

    return topics, coherence_scores

def save_results(topics, coherence_scores, model, k_value):
    """Save analysis results to file."""
    final_likelihood = model.log_likelihood_history[-1]
    with open(f'results/topics_k{k_value}.txt', 'w') as f:
        f.write(f"Topic Modeling Analysis Summary (K={k_value})\n")
        f.write("==============================\n\n")
        f.write(f"Iterations until convergence: {len(model.log_likelihood_history)-1}\n")
        f.write(f"Final log likelihood: {final_likelihood:.6e}\n\n")

        for i, (topic_words, coherence) in enumerate(zip(topics, coherence_scores)):
            f.write(f"\nTopic {i + 1}:\n")
            f.write(f"Words: {', '.join(topic_words)}\n")
            f.write(f"Coherence: {coherence:.4f}\n")

        f.write("\nSummary Statistics:\n")
        f.write(f"Average coherence: {np.mean(coherence_scores):.4f}\n")
        f.write(f"Coherence std dev: {np.std(coherence_scores):.4f}\n")
        f.write(f"Duplicate topics: {sum(1 for score in coherence_scores if score < 0.0001)}\n")

def main():
    print("Loading data...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    print("Preparing data...")
    T, vocabulary = prepare_data_improved(newsgroups.data)
    print(f"Processed data shape: {T.shape}")

    k_values = [10, 20, 30, 50]
    results = {}

    for k in k_values:
        print(f"\nTraining model with K={k}")
        start_time = time.time()

        # Train model
        model = MixtureMultinomialEM(n_topics=k, max_iter=100, tol=1e-4, random_state=42)
        model.fit(T)

        # Analyze results
        topics, coherence_scores = analyze_model(model, vocabulary)

        # Create visualizations
        create_visualizations(model, topics, coherence_scores, k)

        # Save results
        save_results(topics, coherence_scores, model, k)

        # Store results for comparison
        results[k] = {
            'model': model,
            'topics': topics,
            'coherence_scores': coherence_scores,
            'training_time': time.time() - start_time,
            'log_likelihood_history': model.log_likelihood_history
        }

    # Create comparison plot
    plot_k_comparison(results)
    print("\nAnalysis complete. Results saved in plots/ and results/ directories.")

if __name__ == "__main__":
    main()
