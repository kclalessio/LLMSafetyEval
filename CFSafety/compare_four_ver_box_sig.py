import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

models     = ['qwen3', 'llama3', 'gemma3_1b', 'gemma3_4b']
models_to_text = {
    'qwen3': 'Qwen3 1.7B',
    'llama3': 'Llama3.2 3B',
    'gemma3_1b': 'Gemma3 1B',
    'gemma3_4b': 'Gemma3 4B'
}
languages  = ['en', 'it']
languages_to_text = {'en': 'English','it': 'Italian'}
versions   = ['base', 'PE', 'sft-en', 'sft-mix']
versions_to_text = {'base': 'Base','PE': 'Prompt Eng','sft-en': 'SFT (English)','sft-mix': 'SFT (Mixed)'}
categories = [
    'SocialBias', 'Criminal_Unethical', 'Insults_SensitiveTopics',
    'PrivacyLeaks', 'Misleading', 'ScenarioEmbedding_Persuasion'
]
categories_to_text = {
    'SocialBias': 'Social Bias',
    'Criminal_Unethical': 'Criminal and Unethical Content',
    'Insults_SensitiveTopics': 'Insults and Sensitive Topics',
    'PrivacyLeaks': 'Data and Privacy Leaks',
    'Misleading': 'Misleading Content',
    'ScenarioEmbedding_Persuasion': 'Scenario Embedding and Persuasion'
}
color_map_en = {
    'base':   '#1F78B4',
    'PE':     '#33A02C',
    'sft-en': '#FF7F00',
    'sft-mix':'#E31A1C'
}
color_map_it = {
    'base':   '#6BAED6',
    'PE':     '#74C476',
    'sft-en': '#FD8D3C',
    'sft-mix':'#FB6A4A'
}

path_templates = {
    'base':    './{model}/{model}_evaluation_{lang}/detailed_results.csv',
    'PE':      './{model}/PE_{model}_evaluation_{lang}/detailed_results.csv',
    'sft-en':  './{model}/safe_{model}_en_evaluation_{lang}/detailed_results.csv',
    'sft-mix': './{model}/safe_{model}_mix_evaluation_{lang}/detailed_results.csv'
}

output_dir = 'plots/version_comparison_with_box_signif'
os.makedirs(output_dir, exist_ok=True)

def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

raw = {lang: {'overall': {}, 'category': {}} for lang in languages}
for lang in languages:
    for model in models:
        raw[lang]['overall'][model]  = {v: [] for v in versions}
        raw[lang]['category'][model] = {cat: {v: [] for v in versions} for cat in categories}
        for ver in versions:
            path = path_templates[ver].format(model=model, lang=lang)
            df = pd.read_csv(path)
            sub = df[df['category'].isin(categories)]
            raw[lang]['overall'][model][ver] = sub['final_score'].dropna().tolist()
            for cat in categories:
                raw[lang]['category'][model][cat][ver] = (sub.loc[sub['category']==cat, 'final_score'].dropna().tolist())

# ----- plot -----
for lang in languages:
    version_colors = color_map_it if lang == 'it' else color_map_en

    # Overall
    stats_overall = {
        m: {
            v: {
                'mean': np.mean(raw[lang]['overall'][m][v]) if raw[lang]['overall'][m][v] else 0,
            }
            for v in versions
        }
        for m in models
    }

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10,6))

    rects_by_version = {}

    for i, v in enumerate(versions):
        data = [raw[lang]['overall'][m][v] for m in models]
        pos = x + (i - 1.5) * width
        # boxplot
        bplot = ax.boxplot(
            data,
            positions=pos,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=version_colors[v], edgecolor='white'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color=version_colors[v]),
            capprops=dict(color=version_colors[v]),
            flierprops=dict(marker='x', markersize=5, markeredgecolor='lightgray', markerfacecolor='none'),
        )
        
        rects_by_version[v] = pos
        # add version to legend
        if i == 0:
            ax.plot([], [], color=version_colors[v], label=versions_to_text[v])
            
        # mean label and marker
        for m in models:
            i = models.index(m)
            mean = stats_overall[m][v]['mean']
            ax.scatter(pos[i], mean, marker='D', color='black', zorder=5)
            ax.text(pos[i], mean + 0.07,
                    f"{mean:.1f}",
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(facecolor=(1, 1, 1, 0.5), boxstyle='round,pad=0.3', edgecolor='none'))
        
    # significance brackets
    comparisons = [
        ('base','PE'),
        ('PE','sft-en'),
        ('sft-en','sft-mix'),
        ('base','sft-en'),
        ('PE','sft-mix'),
        ('base','sft-mix')
    ]
    y_dict = {
        ('base','PE'):       -0.2,
        ('PE','sft-en'):     -0.2,
        ('sft-en','sft-mix'): -0.2,
        ('base','sft-en'):   -0.4,
        ('PE','sft-mix'):    -0.6,
        ('base','sft-mix'):  -0.8
    }

    hpad = width * 0.05
    for v1, v2 in comparisons:
        y = y_dict[(v1, v2)]
        for idx, m in enumerate(models):
            x1 = rects_by_version[v1][idx] + hpad
            x2 = rects_by_version[v2][idx] - hpad

            arr1 = raw[lang]['overall'][m][v1]
            arr2 = raw[lang]['overall'][m][v2]
            # p = ttest_rel(arr1, arr2).pvalue
            p = wilcoxon(arr1, arr2).pvalue
            stars = p_to_stars(p)
            if stars == 'ns':
                continue
            ax.plot([x1, x2], [y, y], 'k-', lw=0.7)
            ax.plot([x1, x1], [y, y+0.08], 'k-', lw=0.7)
            ax.plot([x2, x2], [y, y+0.08], 'k-', lw=0.7)
            ax.text((x1 + x2)/2, y+0.05, stars,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none"))

    ax.set_title(f'All Versions Comparison in {languages_to_text[lang]} for All Models', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(list(models_to_text.values()))
    ax.set_ylabel('Mean Final Safety Score')
    ax.set_ylim(-0.999,5.7)
    ax.legend(
        handles=[
        *[Patch(facecolor=version_colors[v], label=versions_to_text[v]) for v in versions],
        Line2D([0], [0], marker='D', color='black', label='Mean', markersize=6, linestyle='None'),
        Line2D([0], [0], color='black', lw=1, label='Median'),
        Line2D([0], [0], marker='x', color='lightgray', label='Outliers', markersize=5, linestyle='None')
        ],
        loc='upper left',
        ncol=7,
        fontsize=8.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)

    fn = os.path.join(output_dir, f'overall_by_model_{lang}.png')
    fig.savefig(fn, bbox_inches='tight')
    plt.close(fig)
    print(f"Overall saved")


    # each model
    for model in models:
        labels = ['Overall'] + categories
        n = len(labels)
        x2 = np.arange(n, dtype=float)
        x2[1:] += 0.4

        stats = {'Overall': {}}
        for v in versions:
            data = raw[lang]['overall'][model][v]
            stats['Overall'][v] = {
                'mean': np.mean(data) if data else 0,
            }
        for cat in categories:
            stats[cat] = {}
            for v in versions:
                data = raw[lang]['category'][model][cat][v]
                stats[cat][v] = {
                    'mean': np.mean(data) if data else 0,
                }

        fig, ax = plt.subplots(figsize=(12,6))
        widths = np.array([0.3] + [0.2]*(n-1))
        bar_positions = {v: [] for v in versions}

        for i, v in enumerate(versions):
            data = []
            for lbl in labels:
                if lbl == 'Overall':
                    data.append(raw[lang]['overall'][model][v])
                else:
                    data.append(raw[lang]['category'][model][lbl][v])
            xpos = x2 + (i - 1.5) * widths
            bar_positions[v] = xpos
            # boxplot
            bplot = ax.boxplot(
                data,
                positions=xpos,
                widths=widths,
                patch_artist=True,
                boxprops=dict(facecolor=version_colors[v], edgecolor='white'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color=version_colors[v]),
                capprops=dict(color=version_colors[v]),
                flierprops=dict(marker='x', markersize=5, markeredgecolor='lightgray', markerfacecolor='none'),
            )
            
            # mean label and marker
            for lbl_idx, lbl in enumerate(labels):
                mean_val = stats[lbl][v]['mean']
                ax.scatter(xpos[lbl_idx], mean_val, marker='D', color='black', zorder=5)
                ax.text(xpos[lbl_idx], mean_val + 0.07,
                        f"{mean_val:.1f}",
                        ha='center', va='bottom', fontsize=7,
                        bbox=dict(facecolor=(1, 1, 1, 0.5), boxstyle='round,pad=0.2', edgecolor='none'))
            
            # add version to legend
            if i == 0:
                ax.plot([], [], color=version_colors[v], label=versions_to_text[v])

        hpad = widths[0] * 0.05
        for v1, v2 in comparisons:
            y = y_dict[(v1, v2)]
            for lbl_idx, lbl in enumerate(labels):
                arr1 = (raw[lang]['overall'][model][v1]
                        if lbl=='Overall'
                        else raw[lang]['category'][model][lbl][v1])
                arr2 = (raw[lang]['overall'][model][v2]
                        if lbl=='Overall'
                        else raw[lang]['category'][model][lbl][v2])
                # p = ttest_rel(arr1, arr2).pvalue
                p = wilcoxon(arr1, arr2).pvalue
                stars = p_to_stars(p)
                if stars == 'ns':
                    continue
                x1 = bar_positions[v1][lbl_idx] + hpad
                x2p = bar_positions[v2][lbl_idx] - hpad

                # significance bracket
                ax.plot([x1, x2p], [y, y], 'k-', lw=0.7)
                ax.plot([x1, x1], [y, y+0.08], 'k-', lw=0.7)
                ax.plot([x2p, x2p], [y, y+0.08], 'k-', lw=0.7)
                ax.text((x1 + x2p)/2, y+0.05, stars,
                        ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none"),
                        zorder=6)


        ax.set_title(f'All Versions Comparison in {languages_to_text[lang]} for Model: {models_to_text[model]}', fontsize=14)
        label_texts = ['Overall'] + list(categories_to_text.values())
        ax.set_xticks(x2)
        ax.set_xticklabels(label_texts, rotation=15, ha='right')
        for lbl in ax.get_xticklabels()[:1]:
            lbl.set_fontweight('bold')
        ax.set_ylabel('Mean Final Safety Score')
        ax.set_ylim(-0.999,5.7)
        ax.legend(
            handles=[
            *[Patch(facecolor=version_colors[v], label=versions_to_text[v]) for v in versions],
            Line2D([0], [0], marker='D', color='black', label='Mean', markersize=6, linestyle='None'),
            Line2D([0], [0], color='black', lw=1, label='Median'),
            Line2D([0], [0], marker='x', color='lightgray', label='Outliers', markersize=5, linestyle='None')
            ],
            loc='upper left',
            ncol=7,
            fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linewidth=0.5)
        plt.subplots_adjust(bottom=0.3)

        fn2 = os.path.join(output_dir, f'{model}_breakdown_{lang}.png')
        fig.savefig(fn2, bbox_inches='tight')
        plt.close(fig)
        print(f"{fn2} saved")
