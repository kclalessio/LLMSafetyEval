import pandas as pd
import numpy as np
from scipy import stats
import sys

def tost_equivalence(diff, delta, alpha=0.005):
    n = len(diff)
    mean_diff = diff.mean()
    sd = diff.std(ddof=1)
    se = sd / np.sqrt(n)
    p_low = 1 - stats.t.cdf((mean_diff + delta) / se, df=n-1)
    p_high = stats.t.cdf((mean_diff - delta) / se, df=n-1)
    return (p_low < alpha) and (p_high < alpha)

def validate_proxy(path: str):
    df = pd.read_csv(path)
    configs = [
        ('yn',  '0-1', 'yn_score_api',  'yn_score',   0.025, 0.5),
        ('five', '1-5', 'five_score_api', 'five_score',  0.25,  3.0),
        ('final','0-5', 'final_score_api','final_score', 0.25,  2.5),
    ]

    rows = []
    for name, scale, gold_col, proxy_col, delta, thresh in configs:
        g = df[gold_col].to_numpy()
        p = df[proxy_col].to_numpy()

        # Pearson correlation
        pearson_r, pvalue_pearson = stats.pearsonr(p, g)
        if pvalue_pearson < sys.float_info.min:
            pvalue_pearson = f"< {sys.float_info.min:.1e}"
        
        # Spearman correlation
        spearman_r, pvalue_spearman = stats.spearmanr(p, g)
        if pvalue_spearman < sys.float_info.min:
            pvalue_spearman = f"< {sys.float_info.min:.1e}"

        # Equivalence test
        equiv = tost_equivalence(p - g, delta)

        # Binary accuracy
        acc = ((g >= thresh) == (p >= thresh)).mean()

        rows.append({
            'score_type':       name,
            'scale':            scale,
            'pearson':          pearson_r,
            'pvalue_pearson':   pvalue_pearson,
            'spearman':         spearman_r,
            'pvalue_spearman':  pvalue_spearman,
            'delta':            delta,
            'tost_equivalent':  equiv,
            'binary_accuracy':  acc
        })

    res = pd.DataFrame(rows)
    res.to_csv("proxy_validation.csv", index=False)
    print(res)
    return res

if __name__ == "__main__":
    validate_proxy("gemma3_1b/gemma3_1b_evaluation_en__api/detailed_results_compare.csv")
