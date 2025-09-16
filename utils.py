import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def save_model(model, file_path):
    """Save a model to a file"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """Load a model from a file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_results(rf_history, nn_history, rf_metrics, nn_metrics):
    """Plot training results and evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training history for neural network
    if nn_history and 'accuracy' in nn_history:
        axes[0, 0].plot(nn_history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(nn_history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Neural Network Training History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
    
    # Plot confusion matrices
    cm_rf = rf_metrics['confusion_matrix']
    cm_nn = nn_metrics['confusion_matrix']
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('Random Forest Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Neural Network Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Plot ROC curves
    fpr_rf, tpr_rf, _ = roc_curve(rf_metrics['probabilities'], rf_metrics['probabilities'])
    fpr_nn, tpr_nn, _ = roc_curve(nn_metrics['probabilities'], nn_metrics['probabilities'])
    
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    
    axes[1, 1].plot(fpr_rf, tpr_rf, color='darkorange', lw=2, 
                   label='Random Forest ROC (AUC = %0.2f)' % roc_auc_rf)
    axes[1, 1].plot(fpr_nn, tpr_nn, color='green', lw=2, 
                   label='Neural Network ROC (AUC = %0.2f)' % roc_auc_nn)
    axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('Receiver Operating Characteristic')
    axes[1, 1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.show()