import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from typing import Optional


def plot_confusion_matrix(
        y_true,
        y_pred,
        cmap='YlGn',
        colorbar=False,
        figure_title=False,
        save_figure=False,
        save_figure_filename='Confusion_Matrix'
        ):
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    class_names = list(np.unique(y_true))
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=cmap, ax=ax, colorbar=colorbar)
    
    if figure_title:
        ax.set_title(figure_title)
    
    if save_figure:
        plt.savefig(f"{save_figure_filename}.pdf", format="pdf", dpi=300, transparent=False, bbox_inches="tight")
    
    plt.show()


def plot_feature_importance(
        model,
        df: pd.DataFrame,
        features_columns: list,
        title: Optional[str] = None,
        figsize: tuple = (8, 6),
        cmap: str = 'viridis',
        reverse_cmap: bool = False,
        save_figure: bool = False,
        save_figure_filename='Feature_Importance',
        importance_type: str = 'weight',
        importance_sorting_ascending: bool = False,
        n_top_features: Optional[int] = None
        ):
    
    feature_names = df[features_columns].columns
    feature_importances = model.get_score(importance_type=importance_type)

    importance_df = pd.DataFrame.from_dict(feature_importances, orient='index', columns=['importance'])
    importance_df.index = [feature_names[int(feature[1:])] for feature in importance_df.index]
    importance_df.index.name = 'feature'
    importance_df.reset_index(inplace=True)

    importance_df['importance'] /= importance_df['importance'].sum()
    importance_df['importance_percent'] = importance_df['importance'] * 100
    
    importance_df = importance_df.sort_values(by='importance', ascending=importance_sorting_ascending)

    if n_top_features is not None:
        top_features = importance_df.head(n_top_features)
    else:
        top_features = importance_df
    
    cmap = plt.get_cmap(cmap)

    if reverse_cmap:
        cmap.reversed()

    colors = cmap(np.linspace(0, 1, len(top_features)))

    plt.figure(figsize=figsize)
    bars = plt.barh(top_features['feature'], top_features['importance_percent'], color=colors, zorder=2)
    plt.grid(linestyle='--', axis='x', linewidth=0.85, color='gray', alpha=0.35, zorder=0)
    plt.xlabel('Importance [%]')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()

    for bar, value in zip(bars, top_features['importance_percent']):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2.0, f'{value:.2f} %', va='center', color='darkmagenta')
    
    plt.xlim(0, top_features['importance_percent'].max() + 5)

    plt.subplots_adjust(right=0.85)

    if title:
        plt.title(title)
    
    if save_figure:
        plt.savefig(f"{save_figure_filename}.pdf", format="pdf", dpi=300, transparent=False, bbox_inches="tight")
    
    plt.show()


def plot_xgboost_sweep(
        min_rounds,
        max_rounds,
        step_rounds,
        metric_values,
        figsize=(8, 5),
        title=None,
        x_label='Number of estimators',
        y_label='Score',
        title_size=14,
        x_label_size=12,
        y_label_size=12,
        save_figure=False,
        save_figure_filename='XGBoost_Sweeping'
        ):
    fig, ax = plt.subplots(figsize=figsize)

    full_mean   = np.mean(metric_values, axis=1)
    full_std    = np.std(metric_values, axis=1)

    ax.errorbar(range(min_rounds, max_rounds + step_rounds, step_rounds), full_mean, yerr=full_std, fmt='o', markersize=6, capsize=5, capthick=2, color='C0', ecolor='C3', elinewidth=1, linestyle='')
    ax.grid(True, which='both', linestyle='--', linewidth=0.75, color='gray', alpha=0.5)
    ax.xaxis.set_tick_params(direction='in', which='both', top=True, right=True)
    ax.yaxis.set_tick_params(direction='in', which='both', right=True)
    ax.set_xlabel(x_label, fontsize=x_label_size)
    ax.set_ylabel(y_label, fontsize=y_label_size)

    if title:
        ax.set_title(title, fontsize=title_size)

    if save_figure:
        plt.savefig(f'{save_figure_filename}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
