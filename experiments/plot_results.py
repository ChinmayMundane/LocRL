"""Plot main-paper preference responses from final evaluation records."""
from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from result_data import ROOT, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true', help='Display Figure 2 without saving files.')
    parser.add_argument('--save', action='store_true', help='Save PDF/SVG/PNG (also the default without --show).')
    args = parser.parse_args()
    out = ROOT / 'outputs/plots' 
    data, _, _ = load_data()
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), layout='constrained')
    for method, label, color, marker in [('CAPQL','CAPQL','#0072B2','o'),
            ('PC-SAC','PC-SAC','#D55E00','s'),('Fixed-weight SAC','Fixed SAC','#009E73','^')]:
        g = data[data.method == method].groupby('navigation_weight')
        for ax, key, ylabel in zip(axes, ['efficiency','auc','rmse'],
                                   ['Path efficiency','Covariance AUC','Position RMSE']):
            means = g[key].mean(); sd = g[key].std(ddof=1)
            ax.errorbar(means.index, means, yerr=sd, label=label, color=color,
                        marker=marker, capsize=2, linestyle='--' if method.startswith('Fixed') else '-')
            ax.set_ylabel(ylabel)
        # Landmark visits are stored in the final per-map records.
        values = []
        from result_data import METHODS, SEEDS, tag
        import pandas as pd
        folder, weights = METHODS[method]
        for w in weights:
            visits = [pd.read_csv(ROOT/'results'/folder/f'seed_{seed}'/f'test_maps_{tag(w)}.csv').landmark_visit_fraction.mean() for seed in SEEDS]
            values.append((w, np.mean(visits), np.std(visits, ddof=1)))
        a=np.array(values)
        axes[3].errorbar(a[:,0],a[:,1],yerr=a[:,2],label=label,color=color,marker=marker,capsize=2,
                         linestyle='--' if method.startswith('Fixed') else '-')
    axes[3].set_ylabel('Landmark visit fraction')
    for ax in axes:
        ax.set_xlabel('Navigation weight');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    if args.save or not args.show:
        out.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf','svg','png']:
            fig.savefig(out/f'preference_response.{ext}', dpi=240)
        print(f'Figure 2: {out}')
    if args.show:
        plt.show()
    plt.close(fig)

if __name__=='__main__': main()
