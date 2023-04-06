import uproot as ur
import numpy as np
import sklearn.metrics as skm
import os
import matplotlib.pyplot as plt
import json

def merge_scores(config):
    outfile_list = [ os.path.join(config['output_path'], f'fold_{k}', 'xgboost_output.root') for k in range(config['k_fold']) ]
    
    target_dir =  os.path.join(config['output_path'], 'eval')
    merge_target = os.path.join(target_dir, 'xgboost_output.root')
    if not os.path.exists(target_dir):
        os.makedirs(os.path.dirname(target_dir))

    # Loop over the input files and concatenate the TestTrees
    files = [ur.open(file_name) for file_name in outfile_list]
    with ur.recreate(merge_target) as output:
        output["TestTree"] = ur.concatenate([file['TestTree'] for file in files])
        output["TrainTree"] = ur.concatenate([file['TrainTree'] for file in files])
    return merge_target


def eval_scores(in_data: str, out_d:str = './', do_plot: bool = True, show_plot: bool = False):
    result = {
        'auc': 0.,
        'significance': 0.,
        'mva_score': -999,
    }
    res = ur.open(in_data)

    scores, labels, weight = res['TestTree'].arrays(['scores', 'labels', 'weight'], library='np').values()
    train_scores, train_labels, train_weight = res['TrainTree'].arrays(['scores', 'labels', 'weight'], library='np').values()
    """
    ROC Curve
    """
    fpr, tpr, thresholds = skm.roc_curve(labels, scores, sample_weight=weight, pos_label=1)
    # auc = skm.auc(fpr, tpr)
    auc = np.trapz(tpr, fpr)

    if do_plot:
        mycolors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
        mylinestyles = ['dashed', 'dashed', 'dashed', 'solid']
        mylinewidth = [1.5, 1.5, 1.5, 2.5]
        plots = (fpr, tpr, auc)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)

        plt.plot(
            plots[0], plots[1],
            label=f"AUC = {plots[2]:.3f}",
            color=mycolors[0],
            linestyle=mylinestyles[0],
            linewidth=mylinewidth[0],
        )

        result['auc'] = plots[2]

        plt.ylabel('Signal Eff.')
        plt.xlabel('Background InEff.')
        plt.legend(loc=4)
        if show_plot:
            plt.show()
        fig.savefig(os.path.join(out_d, 'roc.svg'), format='svg')

    """
    Significance Curve
    """
    sig_score = scores[np.where(labels==1)]
    bkg_score = scores[np.where(labels==0)]
    sig_weight, bkg_weight = np.ones(len(sig_score)), np.ones(len(bkg_score))
    if weight is not None:
        sig_weight = weight[np.where(labels==1)]
        bkg_weight = weight[np.where(labels==0)]

    bins = np.linspace(0, 1, num=200, endpoint=True)
    hist_sig, _ = np.histogram(sig_score, bins=bins, weights=sig_weight)
    hist_bkg, _ = np.histogram(bkg_score, bins=bins, weights=bkg_weight)
    s = np.cumsum(hist_sig[::-1])[::-1]
    b = np.cumsum(hist_bkg[::-1])[::-1]


    sig_err = np.sqrt(np.histogram(sig_score, bins=bins, weights=sig_weight ** 2)[0])
    bkg_err = np.sqrt(np.histogram(bkg_score, bins=bins, weights=bkg_weight ** 2)[0])
    s_err = np.sqrt(np.cumsum(sig_err[::-1] ** 2)[::-1])
    b_err = np.sqrt(np.cumsum(bkg_err[::-1] ** 2)[::-1])

    significance = (s / np.sqrt(s + b))
    significance[np.isnan(significance)] = 0

    def sig_unc(s, b, ds, db):
        t1 = ((np.sqrt(s + b) - s / (2 * np.sqrt(s + b))) / (s + b) * ds) ** 2
        t2 = (-(s * 1. / (2 * np.sqrt(s + b)) / (s + b)) * db) ** 2
        return np.sqrt(t1 + t2)

    significance_err = sig_unc(s, b, s_err, b_err)
    significance_err[np.isnan(significance_err)] = 0
    significance_with_min_bkg = max([(y, x) for x, y in enumerate(significance) if b[x] > 1.0])

    result['significance'] = significance_with_min_bkg[0]
    result['significance_no_protect'] = max(significance)
    result['mva_score'] = bins[1 + significance_with_min_bkg[1]]
    result['mva_score_no_protect'] = bins[1 + np.argmax(significance)]


    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
        plt.plot(bins[1:], significance, color='#3776ab')
        plt.fill_between(
            bins[1:], significance - significance_err, significance + significance_err, alpha=0.35,
            edgecolor='#3776ab', facecolor='#3776ab', hatch='///', linewidth=0, interpolate=True
        )

        plt.vlines(
            x=result['mva_score'], ymin=0, ymax=result['significance'],
            colors='purple',
            label=f'max Sig. = {result["significance"]:.3f} at {result["mva_score"]:.2f}'
        )

        plt.ylabel('Significance')
        plt.xlabel('MVA score')
        plt.legend(loc=3)
        if show_plot:
            plt.show()
        fig.savefig(os.path.join(out_d, 'sig.svg'), format='svg')

        """
        Train Test Histogram
        """
        if do_plot:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 10), dpi=80,
                sharex=True, gridspec_kw={'height_ratios': [4, 1]}
            )

            bins = np.linspace(0, 1, num=25, endpoint=True)
            bin_center = (bins[:-1] + bins[1:]) / 2.
            bin_err = bins[1:] - bin_center
            hist_style = dict(
                bins=bins,
                density=True, alpha=0.75, histtype='step', linewidth=2.5
            )

            # for test
            bkg = bkg_score
            sig = sig_score

            hist_sig = np.histogram(sig, bins=bins, weights=sig_weight, density=True)[0]
            hist_bkg = np.histogram(bkg, bins=bins, weights=bkg_weight, density=True)[0]
            sf_s = np.sum(hist_sig) / np.sum(sig)
            sf_b = np.sum(hist_bkg) / np.sum(bkg)
            sig_err = np.sqrt(np.histogram(sig, bins=bins, weights=sig_weight ** 2)[0]) * sf_s
            bkg_err = np.sqrt(np.histogram(bkg, bins=bins, weights=bkg_weight ** 2)[0]) * sf_b
            ax1.errorbar(
                bin_center, hist_sig, yerr=sig_err, xerr=bin_err, color='red', label='sig: test',
                marker='.', alpha=0.75, elinewidth=2, markersize=10, ls='none')
            ax1.errorbar(
                bin_center, hist_bkg, yerr=bkg_err, xerr=bin_err, color='blue', label='bkg: test',
                marker='.', alpha=0.75, elinewidth=2, markersize=10, ls='none')

            # for train
            if train_scores is not None:
                train_bkg = np.where(train_labels==0)
                bkg = train_scores[train_bkg]
                bkg_weight = train_weight[train_bkg]

                train_sig = np.where(train_labels==1)
                sig_weight = train_weight[train_sig]
                sig = train_scores[train_sig]

                s, _, _ = ax1.hist(sig, weights=sig_weight, color='red', label='sig: train', **hist_style)
                b, _, _ = ax1.hist(bkg, weights=bkg_weight, color='blue', label='bkg: train', **hist_style)
                sf_s = np.sum(s) / np.sum(sig)
                sf_b = np.sum(b) / np.sum(bkg)
                s_err = np.sqrt(np.histogram(sig, bins=bins, weights=sig_weight ** 2)[0]) * sf_s
                b_err = np.sqrt(np.histogram(bkg, bins=bins, weights=bkg_weight ** 2)[0]) * sf_b
                ax1.fill_between(
                    bin_center, s - s_err, s + s_err, alpha=0.5, facecolor='red', edgecolor='red',
                    hatch='///', linewidth=0
                )
                ax1.fill_between(
                    bin_center, b - b_err, b + b_err, alpha=0.5, facecolor='blue', edgecolor='blue',
                    hatch='///', linewidth=0
                )

                from scipy import stats
                ks_sig = stats.kstest(hist_sig, s)
                ks_bkg = stats.kstest(hist_bkg, b)

                print('signal: ', ks_sig)
                print('background: ', ks_bkg)

                def divide_unc(a, b, da, db):
                    return np.sqrt((1. / b * da) ** 2 + (-(a / b ** 2) * db) ** 2)

                ratio_s = hist_sig / s
                ratio_b = hist_bkg / b
                ratio_s_unc = divide_unc(hist_sig, s, sig_err, s_err)
                ratio_b_unc = divide_unc(hist_bkg, b, bkg_err, b_err)
                ratio_s[np.isnan(ratio_s)] = 0.0
                ratio_b[np.isnan(ratio_b)] = 0.0
                ratio_s_unc[np.isnan(ratio_s_unc)] = 0
                ratio_b_unc[np.isnan(ratio_b_unc)] = 0

                # ratio plot
                ax2.errorbar(
                    bin_center, ratio_s, yerr=ratio_s_unc, color='red', label='sig: test',
                    marker='o', alpha=0.75, ls='none')
                ax2.errorbar(
                    bin_center, ratio_b, yerr=ratio_b_unc, color='blue', label='bkg: test',
                    marker='o', alpha=0.75, ls='none')
                ax2.set_ylim([0.75, 1.25])
                ax2.axhline(
                    xmin=-1.0, xmax=1.0, y=1.0,
                    c='lightgrey',
                    ls='--'
                )
                ax2.set_ylabel('test / train')

            plt.xlabel('MVA score')
            ax1.legend(loc=1)
            plt.subplots_adjust(wspace=0, hspace=0.1)
            if show_plot:
                plt.show()
            fig.savefig(os.path.join(out_d, 'mva_hist.svg'), format='svg')

            """
            Output JSON
            """
            # Serializing json
            json_object = json.dumps(result, indent=4)

            # Writing to sample.json
            with open(os.path.join(out_d, 'result.json'), "w") as outfile:
                outfile.write(json_object)

    return result

