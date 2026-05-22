import matplotlib.pyplot as plt

datasets = ['NSL-KDD', 'CIC-IDS2017', 'UNSW-NB15']
accuracies = [99.9, 99.6, 96.0]

plt.figure(figsize=(6, 4))
bars = plt.bar(datasets, accuracies, color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylim(90, 100.5)
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy on Different Datasets')
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1, f'{acc:.1f}%', 
             ha='center', va='bottom', color='white', fontsize=11)

plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.show()
