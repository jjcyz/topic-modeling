import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import os

def plot_topic_coherence(topics, coherence_scores):
    """Plot topic coherence scores."""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(1, len(coherence_scores) + 1), coherence_scores)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.xlabel('Topic Number')
    plt.ylabel('Coherence Score')
    plt.title('Topic Coherence Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/topic_coherence_new.png')
    plt.close()

def plot_topic_similarity_heatmap(model):
    """Plot topic similarity matrix."""
    # Compute topic similarity using cosine similarity
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
    plt.title('Topic Similarity Matrix')
    plt.tight_layout()
    plt.savefig('plots/topic_similarity_new.png')
    plt.close()

def plot_topic_evolution(log_likelihood_history):
    """Plot log likelihood evolution during training."""
    plt.figure(figsize=(10, 6))
    plt.plot(log_likelihood_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Log Likelihood')
    plt.title('Model Convergence Over Iterations')

    # Add marker for convergence point
    convergence_point = len(log_likelihood_history) - 1
    final_likelihood = log_likelihood_history[-1]
    plt.plot(convergence_point, final_likelihood, 'ro', markersize=10,
             label=f'Converged at iteration {convergence_point}')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/convergence_evolution_new.png')
    plt.close()

def plot_topic_distinctiveness(model):
    """Visualize topics in 2D space using t-SNE."""
    if model.K < 3:
        print("Warning: Too few topics to create meaningful visualization")
        return

    # Use appropriate perplexity for small number of samples
    perplexity = min(model.K - 1, 5)  # Set perplexity based on number of topics

    try:
        # Apply t-SNE with adjusted parameters
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    n_iter=1000,
                    random_state=42)
        topic_coords = tsne.fit_transform(model.mu)

        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(topic_coords[:, 0], topic_coords[:, 1],
                            c=range(model.K), cmap='tab20', s=100)

        # Add topic numbers and top words
        for i, (x, y) in enumerate(topic_coords):
            plt.annotate(f'Topic {i+1}', (x, y), xytext=(5, 5),
                        textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.7))

        plt.title('Topic Distinctiveness Visualization')
        plt.colorbar(scatter, label='Topic Number')
        plt.savefig('plots/topic_distinctiveness_new.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create topic distinctiveness plot: {str(e)}")

def save_topics_to_file(topics, coherence_scores, model, log_likelihood):
    """Save topics and analysis to a text file."""
    with open('results/topics_new.txt', 'w') as f:
        # Write model summary
        f.write("Topic Modeling Analysis Summary\n")
        f.write("==============================\n\n")

        f.write(f"Number of topics (K): {model.K}\n")
        f.write(f"Iterations until convergence: {len(model.log_likelihood_history)-1}\n")  # Subtract 1 for initial state
        f.write(f"Total iterations including initial state: {len(model.log_likelihood_history)}\n")
        f.write(f"Final log likelihood: {log_likelihood:.6e}\n\n")  # Changed format to scientific notation

        # Write topic details
        f.write("Topics and Coherence Scores:\n")
        f.write("----------------------------\n")
        for i, (topic_words, coherence) in enumerate(zip(topics, coherence_scores)):
            f.write(f"\nTopic {i + 1}:\n")
            f.write(f"Words: {', '.join(topic_words)}\n")
            f.write(f"Coherence: {coherence:.4f}\n")

        # Write summary statistics
        f.write("\nSummary Statistics:\n")
        f.write("------------------\n")
        f.write(f"Average coherence: {np.mean(coherence_scores):.4f}\n")
        f.write(f"Coherence std dev: {np.std(coherence_scores):.4f}\n")
        f.write(f"Number of potential duplicate topics: {sum(1 for score in coherence_scores if score < 0.0001)}\n")
