from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import pandas as pd

def apply_tukey(feature, anova_table, population, df):
    if anova_table['PR(>F)'][0] < 0.05:
        print("Significant ANOVA result, proceed with post-hoc testing.")        
        tukey = pairwise_tukeyhsd(endog=df[feature],    
                                groups=df[population],   
                                alpha=0.05)                  
        return tukey
    else:
        print("No significant differences found by ANOVA.")
        return None
    

def plot_tukey(tukey_results):
    tukey_summary = pd.DataFrame(data=tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])

    fig, ax = plt.subplots(figsize=(12, 8))
    errors = [tukey_summary['meandiff'] - tukey_summary['lower'], tukey_summary['upper'] - tukey_summary['meandiff']]
    significant = tukey_summary['reject'] == True 

    ax.errorbar(tukey_summary['meandiff'], tukey_summary.index, xerr=errors, fmt='none', color='black', ecolor='gray', capsize=5)
    ax.scatter(tukey_summary[significant]['meandiff'], tukey_summary[significant].index, color='red', s=100, edgecolors='k', label='Significant')
    ax.scatter(tukey_summary[~significant]['meandiff'], tukey_summary[~significant].index, color='blue', s=50, label='Not significant')

    ax.axvline(x=0, linestyle='--', color='grey', alpha=0.75)
    ax.set_yticks(tukey_summary.index)
    ax.set_yticklabels([f'{row["group1"]} vs. {row["group2"]}' for index, row in tukey_summary.iterrows()])
    ax.set_xlabel('Mean Difference')
    ax.set_title('Tukey HSD Test Results')
    ax.legend()