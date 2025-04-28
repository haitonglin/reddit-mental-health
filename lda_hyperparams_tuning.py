'''
Hyperparameter tuning to find the best alphas, etas
- K-fold validation, loop over combinations of hyperparameters
- Train LDA on training corpus
- Evaluate on held-out test corpus 
- Aggregate perplexity across folds
- Keep track of the best model

skipped here in this project because it's taking too long, script is just here for reference:

'''
def hyperparams_tuning(corpus,dictionary):
    kf = KFold(n_splits=10)

    # Define possible values for each parameter
    alphas = ['symmetric', 'asymmetric', 'auto']
    etas = ['symmetric', 'auto']
    num_topics = 10

    # Generate all combinations of parameters
    param_grid = product(alphas, etas, [num_topics])

    best_model = None
    best_perplexity = float('inf')  # Initialize with infinity
    best_alpha = None
    best_eta = None

    # Manually perform grid search
    for params in param_grid:
        alpha, eta, num_topics = params

        total_perplexity = 0.0
    
        for train_index, test_index in kf.split(corpus):
            # Split data into training and validation sets
            train_corpus = [corpus[i] for i in train_index]
            test_corpus = [corpus[i] for i in test_index]

            lda_model = LdaModel(corpus=train_corpus, id2word=dictionary,
                                alpha=alpha, eta=eta, num_topics=num_topics,
                                random_state=42)
            perplexity = lda_model.log_perplexity(test_corpus)
            total_perplexity += perplexity

        # Average perplexity across all folds
        avg_perplexity = total_perplexity / 10

        print(f"Params - alpha: {alpha}, eta: {eta}, num_topics: {num_topics}")
        print(f"Average Perplexity: {avg_perplexity:.4f}")
        print()

        if avg_perplexity < best_perplexity:
            best_model = lda_model
            best_perplexity = avg_perplexity
            best_alpha = alpha
            best_eta = eta

    # Print the best model and its perplexity score
    print("Best Model:", best_model)
    print("Best Perplexity Score:", best_perplexity)
