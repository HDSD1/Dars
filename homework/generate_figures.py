#!/usr/bin/env python3
"""
基于K-近邻方法的模式分类研究 —— 实验图生成代码
Generate all experimental figures for the KNN pattern classification paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Output directory: figures/ next to this script
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and prepare dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# Figure 1: KNN Classification Visualization (2D PCA projection)
# ============================================================
def fig1_knn_decision_boundary():
    """KNN分类决策边界可视化"""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    k_values = [1, 5, 15]
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00AA00', '#0000FF'])
    markers = ['o', 's', '^']
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_2d, y)
        
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        for i, marker in enumerate(markers):
            mask = y == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap_bold.colors[i]],
                      marker=marker, edgecolors='k', s=30, linewidth=0.5,
                      label=target_names[i])
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_title(f'K = {k}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_knn_decision_boundary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")

# ============================================================
# Figure 2: K value vs. Accuracy (Cross Validation)
# ============================================================
def fig2_k_vs_accuracy():
    """K值对分类准确率的影响"""
    k_range = range(1, 31)
    cv_scores_mean = []
    cv_scores_std = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
        cv_scores_mean.append(scores.mean())
        cv_scores_std.append(scores.std())
    
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, cv_scores_mean, 'b-o', markersize=4, linewidth=1.5, label='Mean Accuracy')
    ax.fill_between(k_range, 
                    cv_scores_mean - cv_scores_std,
                    cv_scores_mean + cv_scores_std, 
                    alpha=0.2, color='blue')
    
    best_k = np.argmax(cv_scores_mean) + 1
    best_acc = cv_scores_mean[best_k - 1]
    ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best K={best_k} (Acc={best_acc:.4f})')
    ax.scatter([best_k], [best_acc], c='red', s=100, zorder=5, edgecolors='darkred')
    
    ax.set_xlabel('K Value', fontsize=12)
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax.set_title('Effect of K Value on Classification Accuracy (10-fold CV)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 31, 2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_k_vs_accuracy.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved. Best K={best_k}, Accuracy={best_acc:.4f}")

# ============================================================
# Figure 3: Distance Metric Comparison
# ============================================================
def fig3_distance_metric_comparison():
    """不同距离度量对分类结果的影响"""
    metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    metric_labels = ['Euclidean', 'Manhattan', 'Chebyshev', 'Minkowski(p=3)']
    k_range = range(1, 21)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers_list = ['o', 's', '^', 'D']
    
    for metric, label, color, marker in zip(metrics, metric_labels, colors, markers_list):
        scores = []
        for k in k_range:
            if metric == 'minkowski':
                knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=3)
            else:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            cv_score = cross_val_score(knn, X_scaled, y, cv=10).mean()
            scores.append(cv_score)
        ax.plot(k_range, scores, color=color, marker=marker, markersize=4, 
                linewidth=1.5, label=label)
    
    ax.set_xlabel('K Value', fontsize=12)
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax.set_title('Comparison of Distance Metrics', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 21, 2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_distance_metrics.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")

# ============================================================
# Figure 4: Comparison with Other Classification Methods
# ============================================================
def fig4_method_comparison():
    """与其他分类方法的比较"""
    classifiers = {
        'KNN(K=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN(K=9)': KNeighborsClassifier(n_neighbors=9),
        'Bayes': GaussianNB(),
        'Fisher LDA': LinearDiscriminantAnalysis(),
        'SVM(RBF)': SVC(kernel='rbf', C=1.0),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
        results[name] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]
    
    colors = ['#1f77b4', '#4a98d4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.85)
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classification Method Comparison (10-fold CV on Iris)', fontsize=13)
    ax.set_ylim(0.85, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_method_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved.")
    
    # Print table data
    print("\n=== Method Comparison Results ===")
    for name in names:
        print(f"{name}: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")

# ============================================================
# Figure 5: Confusion Matrix for KNN
# ============================================================
def fig5_confusion_matrix():
    """KNN混淆矩阵"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=target_names,
           yticklabels=target_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title=f'KNN (K=5) Confusion Matrix\nAccuracy = {acc:.4f}')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure 5 saved. Test accuracy: {acc:.4f}")

# ============================================================
# Figure 6: Weighted vs Unweighted KNN
# ============================================================
def fig6_weighted_knn():
    """加权KNN与普通KNN的比较"""
    k_range = range(1, 26)
    
    uniform_scores = []
    distance_scores = []
    
    for k in k_range:
        knn_u = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        knn_d = KNeighborsClassifier(n_neighbors=k, weights='distance')
        
        score_u = cross_val_score(knn_u, X_scaled, y, cv=10).mean()
        score_d = cross_val_score(knn_d, X_scaled, y, cv=10).mean()
        
        uniform_scores.append(score_u)
        distance_scores.append(score_d)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, uniform_scores, 'b-o', markersize=4, linewidth=1.5, label='Uniform Weight')
    ax.plot(k_range, distance_scores, 'r-s', markersize=4, linewidth=1.5, label='Distance Weight')
    
    ax.set_xlabel('K Value', fontsize=12)
    ax.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax.set_title('Uniform vs Distance-Weighted KNN', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 26, 2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_weighted_knn.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 6 saved.")


if __name__ == '__main__':
    print("="*50)
    print("Generating all experimental figures...")
    print("="*50)
    fig1_knn_decision_boundary()
    fig2_k_vs_accuracy()
    fig3_distance_metric_comparison()
    fig4_method_comparison()
    fig5_confusion_matrix()
    fig6_weighted_knn()
    print("\n" + "="*50)
    print("All figures generated successfully!")
    print("="*50)
