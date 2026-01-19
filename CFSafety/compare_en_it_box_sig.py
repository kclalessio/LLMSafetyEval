import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

models    = ['qwen3', 'llama3', 'gemma3_1b', 'gemma3_4b']
models_to_text= {
    'qwen3': 'Qwen3 1.7B',
    'llama3': 'Llama3.2 3B',
    'gemma3_1b': 'Gemma3 1B',
    'gemma3_4b': 'Gemma3 4B'
}
languages = ['en', 'it']
lang_to_text = {'en': 'English','it': 'Italian'}
categories = [
    'SocialBias',
    'Criminal_Unethical',
    'Insults_SensitiveTopics',
    'PrivacyLeaks',
    'Misleading',
    'ScenarioEmbedding_Persuasion'
]
categories_to_text = {
    'SocialBias': 'Social Bias',
    'Criminal_Unethical': 'Criminal and Unethical Content',
    'Insults_SensitiveTopics': 'Insults and Sensitive Topics',
    'PrivacyLeaks': 'Data and Privacy Leaks',
    'Misleading': 'Misleading Content',
    'ScenarioEmbedding_Persuasion': 'Scenario Embedding and Persuasion'
}
color_map = {
    'en': '#1F78B4',
    'it': '#6BAED6'
}

output_dir = 'plots/eval_en_it_with_box_signif'
os.makedirs(output_dir, exist_ok=True)

def p_to_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

raw_scores = {
    cat: {model: {'en': [], 'it': []} for model in models}
    for cat in categories
}

for model in models:
    for lang in languages:
        path = f'./{model}/{model}_evaluation_{lang}/detailed_results.csv'
        df = pd.read_csv(path)
        df = df[df['category'].isin(categories)]
        for cat in categories:
            raw_scores[cat][model][lang] = df.loc[df['category'] == cat, 'final_score'].dropna().tolist()

# ----- OVERALL -----
def plot_overall():
    overall = {
        m: {
            'en': sum((raw_scores[cat][m]['en'] for cat in categories), []),
            'it': sum((raw_scores[cat][m]['it'] for cat in categories), [])
        }
        for m in models
    }
    stats = {}
    for m, data in overall.items():
        en = np.array(data['en'])
        it = np.array(data['it'])
        stats[m] = {
            'en_mean': np.mean(en) if en.size else 0,
            # 'en_std':  np.std(en, ddof=1) if en.size>1 else 0,
            'it_mean': np.mean(it) if it.size else 0,
            # 'it_std':  np.std(it, ddof=1) if it.size>1 else 0,
            # 'p':       ttest_rel(en, it).pvalue if en.size and it.size else None
            'p':       wilcoxon(en, it).pvalue if en.size and it.size else None
        }
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,6))
    # English boxplots
    data_en = [overall[m]['en'] for m in models]
    bp_en = ax.boxplot(data_en,
                       positions=x - width/2,
                       widths=width,
                       patch_artist=True,
                       boxprops=dict(facecolor=color_map['en'],edgecolor='white'),
                       medianprops=dict(color='black'),
                    #    medianprops=dict(color='white'),
                    #    medianprops=dict(color='gray'),
                        whiskerprops=dict(color=color_map['en']),
                        capprops=dict(color=color_map['en']),   
                       showfliers=False)
    # Italian boxplots
    data_it = [overall[m]['it'] for m in models]
    bp_it = ax.boxplot(data_it,
                       positions=x + width/2,
                       widths=width,
                       patch_artist=True,
                       boxprops=dict(facecolor=color_map['it'], edgecolor='white'),
                       medianprops=dict(color='black'),
                    #    medianprops=dict(color='white'),
                    #    medianprops=dict(color='gray'),
                        whiskerprops=dict(color=color_map['it']),
                        capprops=dict(color=color_map['it']),
                       showfliers=False)


    # mean label and marker
    for i, m in enumerate(models):
        en_mean = stats[m]['en_mean']
        it_mean = stats[m]['it_mean']
        # English diamond
        ax.scatter(x[i] - width/2, en_mean, marker='D', color='black', zorder=5)
        ax.text(x[i] - width/2, en_mean + 0.07, 
                # f"{en_mean:.2f}",
                f"{en_mean:.1f}",
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(facecolor=(1, 1, 1, 0.5), boxstyle='round,pad=0.3', edgecolor='none'))
        # Italian diamond
        ax.scatter(x[i] + width/2, it_mean, marker='D', color='black', zorder=5)
        ax.text(x[i] + width/2, it_mean + 0.07, 
                # f"{it_mean:.2f}",
                f"{it_mean:.1f}",
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(facecolor=(1, 1, 1, 0.5), boxstyle='round,pad=0.3', edgecolor='none'))
    
    # significance brackets
    y0 = -0.25
    for i, m in enumerate(models):
        p = stats[m]['p']
        if p is None: continue
        stars = p_to_stars(p)
        if stars == 'ns': continue
        x1, x2 = x[i] - width/2, x[i] + width/2
        ax.plot([x1, x2], [y0, y0], 'k-', lw=1)
        ax.plot([x1, x1], [y0, y0+0.1], 'k-', lw=1)
        ax.plot([x2, x2], [y0, y0+0.1], 'k-', lw=1)
        ax.text((x1+x2)/2, y0+0.05, stars,
                ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle=
                          "round,pad=0.05", 
                          facecolor="white", 
                          edgecolor="none"))

    ax.set_title('Language Comparison for All Models', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models_to_text.values())
    ax.set_ylim(-0.5, 5.7)
    ax.set_ylabel('Mean Final Safety Score')
    ax.legend(
        handles=[
            Patch(facecolor=color_map['en'], label='English'),
            Patch(facecolor=color_map['it'], label='Italian'),
            Line2D([0], [0], marker='D', color='black', label='Mean', markersize=6, linestyle='None'),
            Line2D([0], [0], color='black', lw=1, label='Median')
        ],
        loc='upper left',
        ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # ax.axhline(0, color='black', linewidth=0.5)
    plt.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(output_dir, 'overall_models.png'), bbox_inches='tight')
    plt.close(fig)
    print("Overall plot saved")



# ----- EACH MODEL -----
def plot_model(model):
    stats = {}
    # overall of model
    all_en = sum((raw_scores[cat][model]['en'] for cat in categories), [])
    all_it = sum((raw_scores[cat][model]['it'] for cat in categories), [])
    stats['Overall'] = {
        'en_mean': np.mean(all_en) if all_en else 0,
        # 'en_std':  np.std(all_en, ddof=1) if len(all_en)>1 else 0,
        'it_mean': np.mean(all_it) if all_it else 0,
        # 'it_std':  np.std(all_it, ddof=1) if len(all_it)>1 else 0,
        # 'p':       ttest_rel(all_en, all_it).pvalue if all_en and all_it else None
        'p':       wilcoxon(all_en, all_it).pvalue if all_en and all_it else None
    }
    
    # each category of model
    for cat in categories:
        en = raw_scores[cat][model]['en']
        it = raw_scores[cat][model]['it']
        stats[cat] = {
            'en_mean': np.mean(en) if en else 0,
            # 'en_std':  np.std(en, ddof=1) if len(en)>1 else 0,
            'it_mean': np.mean(it) if it else 0,
            # 'it_std':  np.std(it, ddof=1) if len(it)>1 else 0,
            # 'p':       ttest_rel(en, it).pvalue if en and it else None
            'p':       wilcoxon(en, it).pvalue if en and it else None
        }

    labels = ['Overall'] + categories
    n = len(labels)
    gap = 0.5
    x = np.arange(n, dtype=float)
    x[1:] += gap
    w_cat = 0.35
    w_overall = 0.5

    fig, ax = plt.subplots(figsize=(12,6))
    for lang, color in color_map.items():
        # heights = [stats[l][f'{lang}_mean'] for l in labels]
        # errs    = [stats[l][f'{lang}_std']  for l in labels]
        widths  = [w_overall] + [w_cat]*(n-1)
        offsets = [(-w/2 if lang=='en' else w/2) for w in widths]
        xpos    = x + np.array(offsets)
        # boxplots
        data_lang = []
        for lbl in labels:
            if lbl == 'Overall':
                all_vals = []
                for cat in categories:
                    all_vals.extend(raw_scores[cat][model][lang])
                data_lang.append(all_vals)
            else:
                data_lang.append(raw_scores[lbl][model][lang])

        bp = ax.boxplot(data_lang,
                        positions=xpos,
                        widths=widths,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, edgecolor='white'),
                        showfliers=False,
                        medianprops=dict(color='black'),
                        # medianprops=dict(color='white'),
                        # medianprops=dict(color='gray'),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        )
        
        # mean label and marker
        for xi, lbl in zip(xpos, labels):
            mean_val = stats[lbl][f'{lang}_mean']
            ax.scatter(xi, mean_val, marker='D', color='black', zorder=5)
            ax.text(xi, mean_val + 0.07, 
                    # f"{mean_val:.2f}",
                    f"{mean_val:.1f}",
                    ha='center', va='bottom', fontsize=9, 
                    bbox=dict(facecolor=(1, 1, 1, 0.5), boxstyle='round,pad=0.3', edgecolor='none'))

    # ax.axhspan(-1, 0, facecolor='white', zorder=2)
    
    # significance brackets
    y0 = -0.25
    for idx, lbl in enumerate(labels):
        p = stats[lbl]['p']
        if p is None: continue
        stars = p_to_stars(p)
        if stars == 'ns': continue
        w = w_overall if idx==0 else w_cat
        x1 = x[idx] - w/2
        x2 = x[idx] + w/2
        ax.plot([x1, x2], [y0, y0], 'k-', lw=1)
        ax.plot([x1, x1], [y0, y0+0.1], 'k-', lw=1)
        ax.plot([x2, x2], [y0, y0+0.1], 'k-', lw=1)
        ax.text((x1+x2)/2, y0+0.05, stars, ha='center', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none"),
                zorder=6)

    ax.set_title(f'Language Comparison for Model: {models_to_text[model]}', fontsize=16)
    ax.set_xticks(x)
    label_texts = ['Overall'] + list(categories_to_text.values())
    ax.set_xticklabels(label_texts, rotation=15, ha='right')
    for lbl in ax.get_xticklabels()[:1]:
        lbl.set_fontweight('bold')
    ax.set_ylabel('Mean Final Safety Score')
    ax.set_ylim(-0.5, 5.7)
    ax.legend(
        handles=[
            Patch(facecolor=color_map['en'], label='English'),
            Patch(facecolor=color_map['it'], label='Italian'),
            Line2D([0], [0], marker='D', color='black', label='Mean', markersize=6, linestyle='None'),
            Line2D([0], [0], color='black', lw=1, label='Median')
        ],
        loc='upper left',
        ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # ax.axhline(0, color='black', linewidth=0.5)
    plt.subplots_adjust(bottom=0.3)
    fn = os.path.join(output_dir, f'{model}_breakdown.png')
    fig.savefig(fn, bbox_inches='tight')
    plt.close(fig)
    print(f"{model} plot saved")

plot_overall()
for m in models:
    plot_model(m)
