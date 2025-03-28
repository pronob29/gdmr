import os
import numpy as np
import corpus
import vocabulary
from gdmr import gDMR  # Ensure this is the correct import path for your gDMR class
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the output directory
output_dir = "gdmr_content_only"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the full datasets
full_corpus = corpus.Corpus.read('cleaned_text.txt')
full_vecs_dataset = corpus.Corpus.read('merged_user_embeddings_dataset.txt', dtype=np.float32)
# Convert the dataset of covariates to a NumPy array
full_vecs = np.array(list(full_vecs_dataset), dtype=np.float32)
voca = vocabulary.Vocabulary()  # Initialize vocabulary
docs = voca.read_corpus(full_corpus)

# Split the processed docs into training and testing sets
docs_train, docs_test = train_test_split(docs, test_size=0.2, random_state=100)
vecs_train, vecs_test = train_test_split(full_vecs, test_size=0.2, random_state=100)

# Create and train the gDMR model
G = 20  # Number of topics
sigma = 1.0  # Hyperparameter for normal distribution in gDMR
beta = 0.01  # Hyperparameter for Dirichlet prior in gDMR
iterations = 1000  # Adjusted for a more realistic number of iterations
target_number = 700  # The iteration number after which to start updating regression parameters
burn_in = 350  # Number of iterations to run after switching to BFGS optimization before recalculating alpha

# Initialize DMR model (starts training as LDA, transitions to DMR)
dmr = gDMR(G, sigma, beta, docs_train, vecs_train, voca.size(), target_number=target_number)
dmr.training(iteration=iterations, voca=voca, burn_in=burn_in)  # Full training, includes DMR updates after target_number

# Save the DMR model
model_file = os.path.join(output_dir, 'gdmr_model.joblib')
dump(dmr, model_file)

# Plot and save the model perplexity over iterations
iteration_range = np.arange(0, iterations, iterations // len(dmr.perplexity_scores))

plt.figure(figsize=(10, 6))
plt.plot(iteration_range, dmr.perplexity_scores, marker='o', linestyle='-', color='b')
plt.title('Model Perplexity Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.grid(True)
perplexity_plot_file = os.path.join(output_dir, 'model_perplexity.png')
plt.savefig(perplexity_plot_file, format='png', dpi=300)
plt.close()

# Do the same for objective function values
if dmr.objective_values:
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(dmr.objective_values)), dmr.objective_values, marker='o', linestyle='-', color='r')
    plt.title('Objective Function Value Across BFGS Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.grid(True)
    objective_plot_file = os.path.join(output_dir, 'objective_function.png')
    plt.savefig(objective_plot_file, format='png', dpi=300)
    plt.close()

# Generate and save the top words and lambda values
top_n_words = 15
top_n_lambdas = 10
output_file = os.path.join(output_dir, 'top_words_and_lambdas.txt')

with open(output_file, 'w') as file:
    word_distributions = dmr.word_dist_with_voca(voca)
    for group in range(G):
        # Top words
        top_words = sorted(word_distributions[group].items(), key=lambda x: x[1], reverse=True)[:top_n_words]
        file.write(f"Group {group}:\nTop Words:\n")
        for word, prob in top_words:
            file.write(f"  {word}: {prob:.4f}\n")

        # Top lambda values
        lambda_values = dmr.Lambda[group]
        top_lambda_indices = np.argsort(-lambda_values)[:top_n_lambdas]
        file.write("Top Lambda Values:\n")
        for idx in top_lambda_indices:
            file.write(f"  Lambda[{idx}]: {lambda_values[idx]:.4f}\n")
        file.write("\n")

print(f"Top words and lambda values per group saved to '{output_file}'")
