
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_hiv_genome(lrtt_df, maf_df, output_file):
        
    #GENE MAP FOR HIV GENOME
    gene_map = [
        {"gene": "gag", "start": 790, "end": 2292, "color": "purple", "row": 1},
        {"gene": "pol", "start": 2253, "end": 5096, "color": "green", "row": 3},
        {"gene": "vif", "start": 5041, "end": 5619, "color": "gray", "row": 1},
        {"gene": "vpr", "start": 5559, "end": 5850, "color": "green", "row": 3},
        {"gene": "tat", "start": 5831, "end": 6045, "color": "beige", "row": 2},
        {"gene": "rev", "start": 5970, "end": 6045, "color": "green", "row": 3},
        {"gene": "vpu", "start": 6062, "end": 6310, "color": "red", "row": 2},
        {"gene": "env (gp120)", "start": 6225, "end": 7758, "color": "blue", "row": 3},
        {"gene": "env (gp41)", "start": 7758, "end": 8795, "color": "brown", "row": 3},
        {"gene": "rev", "start": 8379, "end": 8653, "color": "green", "row": 2},
        {"gene": "nef", "start": 8797, "end": 9417, "color": "purple", "row": 1},
    ]

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 0.5]})

    ## LRTT PLOT
    sns.lineplot(data=lrtt_df, x='xcoord', y='mean_lrtt', hue='TSI_category', palette='Set1', ax=axs[0])
    axs[0].set_ylabel('Mean LRTT')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)

    ## MAF12c PLOT
    sns.lineplot(data= maf_df, x='Window_Centre', y='MAF12c_Mean', hue='TSI_category', palette='Set1', ax=axs[1])
    axs[1].set_ylabel('MAF12c in window')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

    ## MAF3c PLOT
    sns.lineplot(data= maf_df, x='Window_Centre', y='MAF3c_Mean', hue='TSI_category', palette='Set1', ax=axs[2])
    axs[2].set_ylabel('MAF3c in window')
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['left'].set_visible(False)
    axs[2].spines['bottom'].set_visible(False)

    for gene in gene_map:
        y_position = gene["row"] * 0.5
        axs[3].add_patch(patches.Rectangle((gene["start"], y_position), gene["end"] - gene["start"], 0.4, edgecolor='white', facecolor=gene["color"], label=gene["gene"]))

    #REMOVE OUTER LINES OF FIGURE
    axs[3].set_yticks([])
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)
    axs[3].spines['left'].set_visible(False)
    axs[3].spines['bottom'].set_visible(False)

    for gene in gene_map:
        y_position = gene["row"] * 0.5 + 0.2
        axs[3].text((gene["start"] + gene["end"]) / 2, y_position, gene["gene"], horizontalalignment='center', verticalalignment='center')

    axs[3].set_xlabel('Genome Coordinate')
    axs[3].set_xlim([0, 10000])
    axs[3].set_ylim([0, 2])

    for ax in axs[:-1]:
        ax.get_legend().remove()

    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(handles, labels, title='TSI', title_fontsize='10', fontsize='10', loc='center right', bbox_to_anchor=(0.99, 0.5))
    plt.xlabel('Genome Coordinate')
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    #SAVE FIGURE 
    plt.savefig(output_file)
    plt.close(fig)

    return output_file