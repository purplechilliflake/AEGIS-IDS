import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)
import seaborn as sns
import os

class ModelEvaluator:
    """
    Generates comprehensive evaluation visualizations
    and performance metrics
    """
    
    def __init__(self):
        self.model = joblib.load("models/intrusion_detection_rf.pkl")
        self.test = pd.read_csv("data/processed/test.csv")
        self.X_test = self.test.drop(columns=[col for col in ['class', 'label', 'difficulty_level'] if col in self.test.columns])
        self.y_test = self.test['label']
        
    def generate_confusion_matrix(self):
        """Create annotated confusion matrix"""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png')
        plt.close()
    
    def generate_roc_curve(self):
        """Generate ROC curve with AUC"""
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('results/roc_curve.png')
        plt.close()
    
    def generate_feature_importance(self):
        """Plot feature importance"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-20:]
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [self.X_test.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    evaluator = ModelEvaluator()
    print("Generating evaluation visualizations...")
    evaluator.generate_confusion_matrix()
    evaluator.generate_roc_curve()
    evaluator.generate_feature_importance()
    print("✅ Evaluation complete! Check the 'results' directory.")
