import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

class CorrectProbe:
    """correct probe using logistic regression"""
    def __init__(self, max_iter=100, random_state=42, C=0.0005):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=max_iter, 
            random_state=random_state,
            class_weight='balanced', # automatically handle class imbalance
            C = C
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        """train model"""
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        # train logistic regression
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X):
        """predict class"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """predict probability"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]  # return the probability of the positive class

def load_and_prepare_data(data_path):
    """
    load data and prepare training data
    Args:
        data_path: pickle file path
        balance_samples: whether to balance samples, default True
    Returns:
        X: hidden states, shape约为 [num_samples, num_layers, hidden_size]
        y: labels (1: correct, 0: incorrect)
    """
    print(f"Loading data: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data loaded, contains {len(data)} samples")
    
    hidden_states = []
    y = []
    # only process the last 10000 samples (if the number of samples is less than 10000, process all samples)
    data = data[-10000:] if len(data) > 10000 else data
    for i, sample in enumerate(data):
        # question_hs: the hidden state of the last token of each layer
        # expected shape: [num_layers, hidden_size]
        question_hs = np.array(sample['question_hs'])
        hidden_states.append(question_hs)
        y.append(int(sample['correct']))

    # convert to numpy array, shape [num_samples, num_layers, hidden_size]
    X = np.array(hidden_states)
    y = np.array(y)
    
    print(f"Contains {len(X)} training samples, each sample hidden state shape: {X.shape[1:]}")
    
    return X, y

def train_probe(X, y, test_size=0.2, max_iter=1000, fig_path=None):
    """
    train probe    
    """
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print("Starting to train logistic regression model...")
    
    # initialize and train model
    model = CorrectProbe(max_iter=max_iter)
    model.fit(X_train, y_train)
    
    print("Training completed!")
    
    # predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # calculate detailed metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n=== Final evaluation results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    
    # plot simplified results
    plt.figure(figsize=(8, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    values = [accuracy, precision, recall, f1, auc]
    
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylabel('Score')
    plt.title('Logistic Regression Performance Metrics')
    plt.ylim(0, 1)
    
    # add numerical labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'model_type': 'LogisticRegression',
        'feature_importance': model.model.coef_[0] if hasattr(model.model, 'coef_') else None
    }

def save_model_and_results(model, results, model_path, results_path):
    """save model and results"""
    # 保存模型 (使用pickle)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")
    
    # convert numpy array to list for JSON serialization
    results_serializable = results.copy()
    if 'feature_importance' in results_serializable and results_serializable['feature_importance'] is not None:
        results_serializable['feature_importance'] = results_serializable['feature_importance'].tolist()
    
    # save results
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {results_path}")

def main(data_path, output_dir, max_iter=1000):
    """
    main function
    Args:
        max_iter: maximum number of iterations for logistic regression
    """
    print("=== Correct Probe training program (Logistic Regression) ===")
    print(f"Maximum number of iterations: {max_iter}")
    
    X, y = load_and_prepare_data(data_path)
    
    # X: [num_samples, num_layers, hidden_size]
    if X.ndim != 3:
        raise ValueError(f"Expected X to be a 3D array [num_samples, num_layers, hidden_size], but got shape {X.shape}")

    num_samples, num_layers, hidden_size = X.shape
    print(f"Will train probe for each layer, total {num_layers} layers, each layer hidden_size = {hidden_size}")

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for layer_idx in range(num_layers):
        print(f"\n=== Starting to train the probe of layer {layer_idx} ===")
        # get the features of the layer: [num_samples, hidden_size]
        X_layer = X[:, layer_idx, :]

        # train probe
        suffix = f"_layer{layer_idx}_correct"
        base_name = data_path.split("/")[-1].replace(".pkl", "")

        fig_path = f"{output_dir}/{base_name}{suffix}.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)

        model, results = train_probe(X_layer, y, max_iter=max_iter, fig_path=fig_path)
    
        # save model and results
        model_path = f"{output_dir}/{base_name}{suffix}.pkl"
        results_path = f"{output_dir}/{base_name}{suffix}.json"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
        save_model_and_results(model, results, model_path, results_path)
    
        all_results[f"layer_{layer_idx}"] = results

        print(f"\n=== Layer {layer_idx} training completed ===")
        print(f"Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}, AUC: {results['auc']:.4f}")

    print("\n=== All layers probe training completed ===")
    print(f"Total {num_layers} probe models trained")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Correct Probe (Logistic Regression)')
    parser.add_argument('--data_path', type=str, required=True, help='Data path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--max-iter', type=int, default=1000, help='Maximum number of iterations for logistic regression (default: 1000)')
    
    args = parser.parse_args()
    
    main(data_path=args.data_path,
         output_dir=args.output_dir,
         max_iter=args.max_iter)
