import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_data_distribution(data, title, xmin=None, xmax=None, ymin=None, ymax=None):
    if isinstance(data, pd.DataFrame):
        data = data.values
    plt.figure(figsize=(4, 7))
    plt.subplot(211)
    sns.histplot(data.flatten(), bins=100)
    if (xmin is not None) or (xmax is not None):
        plt.xlim(xmin, xmax)
    plot_xlim = plt.gca().get_xlim()
    plt.title('All Values')
    plt.subplot(212)
    sns.histplot(data.flatten()[data.flatten() > 0], bins=100)
    plt.xlim(plot_xlim)
    plt.title('Non-Zero Values')
    plt.suptitle("Distribution of " + title, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    ncol = 3
    nrow = 1
    fig = plt.figure(figsize=(4 * ncol, 3 * nrow))

    plt.subplot(nrow, ncol, 1)
    plt.plot(history['train_f1_score'], '-', label=f'Train', color='blue', alpha=0.5)
    plt.plot(history['val_f1_score'], '--', label=f'Validation', color='red', alpha=0.5)
    plt.title( 'F1 score')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(nrow, ncol,  2 )
    plt.plot(history['train_presence_loss'], '-', label=f'Train', color='blue', alpha=0.5)
    plt.plot(history['val_presence_loss'], '--', label=f'Validation', color='red', alpha=0.5)
    plt.title('Presence Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(nrow, ncol,  3)
    plt.plot(history['train_nonzero_loss'], '-', label=f'Train', color='blue', alpha=0.5)
    plt.plot(history['val_nonzero_loss'], '--', label=f'Validation', color='red', alpha=0.5)
    plt.title('Non-zero Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    return fig



def plot_reconstructed_distribution(X_test_presence, X_test_nonzero, f1, masked_mse_test, soft_masked_mse_test, reconstructed):
    ncol = 3
    nrow = 1

    plt.figure(figsize=(5 * ncol, 3 * nrow))

    plt.subplot(nrow, ncol, 1)
    sns.histplot(X_test_presence.flatten(), bins=10)
    plt.title( 'Reconstructed Presence ')
    plt.text(0.5, plt.ylim()[1] * 0.8, f'F1 score = {f1:.2f}', fontsize=14)

    plt.subplot(nrow, ncol, 2)
    sns.histplot(X_test_nonzero.flatten(), bins=100)
    plt.text(0.5, plt.ylim()[1] * 0.8, f'masked MSE = {masked_mse_test:.3f}', fontsize=14, ha='center')
    plt.title('Reconstructed non zero Values ')

    plt.subplot(nrow, ncol, 3)
    sns.histplot(reconstructed.flatten(), bins=100)
    plt.ylim(0, 10000)
    plt.text(0.1, plt.ylim()[1] * 0.8, f'soft masked MSE = {soft_masked_mse_test:.3f}', fontsize=14)
    plt.title('Reconstructed Values ')

    plt.tight_layout()



def plot_confusion_matrix(y_true, y_pred):

    # Generate confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix with proportions
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Set Presence/Absence Reconstruction')
    plt.tight_layout()
    plt.show()


def plot_auc(y_true, y_pred):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)
    ap = average_precision_score(y_true, y_pred)
    

    # Plot ROC and PRC curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.2f}')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")

    # PRC curve
    ax2.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.2f}\nAUPRC = {auprc:.2f}')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")

    plt.tight_layout()
    return fig



def scatter_plot_nonzero(X_test, recon_nonzero, softmask, soft_masked_mse_test, x_true_masked_mse_test):

    plt.figure(figsize=(15, 4))

    plt.subplot(131)
    plt.scatter(X_test, recon_nonzero, alpha=0.1, s=1)  # evaluation of decoder2
    plt.plot([X_test.min(), X_test.max()], [X_test.min(), X_test.max()], 'r--', lw=2)
    plt.xlabel('X_pred')
    plt.ylabel('recon_nonzero')
    plt.title('reconstructed value by decoder 2')
    # plt.text(0.05, 0.95, f'softly masked MSE: {soft_masked_mse_test:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.subplot(132)
    plt.scatter(X_test.flatten()[X_test.flatten() > 0], recon_nonzero.flatten()[X_test.flatten() > 0], alpha=0.1, s=1)  # on only the nonzero values 
    plt.plot([X_test.min(), X_test.max()], [X_test.min(), X_test.max()], 'r--', lw=2)
    plt.xlabel('X_test')
    plt.ylabel('X_pred')
    plt.title('reconstructed value filtered by true presence')
    plt.text(0.05, 0.95, f'Masked MSE: {x_true_masked_mse_test:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


    plt.subplot(133)
    plt.scatter(X_test.flatten(), (recon_nonzero * softmask).flatten(), alpha=0.1, s=1)  # loss for training phase 2
    plt.plot([X_test.min(), X_test.max()], [X_test.min(), X_test.max()], 'r--', lw=2)
    plt.xlabel('X_pred')
    plt.ylabel('recon_nonzero')
    plt.title('reconstructed value filtered by softmask')
    plt.text(0.05, 0.95, f'Softly Masked MSE: {soft_masked_mse_test:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


    plt.tight_layout()
    plt.show()


# add vectors to the PCA plot to show the influence of each feature
def add_vectors(df_pca, pca, label_1, label_2, features, top=5):
    # Calculate the impact of each feature on the PCA components
    impact = np.sum(np.abs(pca.components_), axis=0)

    # Get the indices of the features with the most impact
    top_indices = np.argsort(impact)[-top:]
    # Get the names of the top features
    top_features = [features[i] for i in top_indices]

    for i, feat in enumerate(features):
            if feat in top_features:
                    plt.arrow(0, 0, pca.components_[0, i]*max(df_pca[label_1]), pca.components_[1, i]*max(df_pca[label_2]), 
                            color='black', width=0.006, head_width=0.03, linestyle=':')
                    plt.text(pca.components_[0, i]*max(df_pca[label_1])*1.05, pca.components_[1, i]*max(df_pca[label_2])*1.05, 
                            feat, color='black', size = 8)
                    

def plot_latents(df_results, col_factors, reduced_method='pca', show_vectors=False):
    ncol = 4
    features = [feat for feat in col_factors if feat != 'country']
    nrow = len(features) // ncol + (len(features) % ncol > 0)
    plt.figure(figsize=(4 * ncol, 3 * nrow))
    for i, factor in enumerate(features):
        plt.subplot(nrow, ncol, i+1)
        if df_results[factor].dtype == 'object':
            palette = sns.color_palette("Set1", len(df_results[factor].unique()))
        else:
            palette = 'viridis'
        sns.scatterplot(x=f'{reduced_method}_latent_dim_1', y=f'{reduced_method}_latent_dim_2', hue=factor, data=df_results, s=4, alpha=1, legend='brief', palette=palette)
        plt.title(f'{reduced_method.upper()}  {factor}')
        plt.xlabel(f'{reduced_method}_latent_dim_1')
        plt.ylabel(f'{reduced_method}_latent_dim_2')
        if show_vectors: 
            add_vectors( df_results, pca_original, f'{reduced_method}_latent_dim_1', f'{reduced_method}_latent_dim_2', df_genus.columns, top=5)
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

