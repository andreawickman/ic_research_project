import pandas as pd 
import numpy as np

def compute_MAF(df, codon_pos_one_two, codon_pos_three, tsi_intervals, tsi_strata, output_file):
    
    df['TSI_category'] = pd.Categorical(
        np.select(tsi_intervals, tsi_strata),
        categories=tsi_strata, 
        ordered=True
    )
    #window-level specifications
    start_pos = 950
    end_pos = 9650
    window_size = 250
    shift = 25
    results_list = []

    for start in range(start_pos, end_pos - window_size + 1, shift):
        #end coordinate of the window
        end = start + window_size
        #center of window
        centre = (start + end)/2
        #coordinates of each window 
        window_range = list(map(str, range(start, end)))

        #store the coordinate positions that are relevant to each MAF category 
        relevant_12c = [pos for pos in window_range if pos in codon_pos_one_two]
        relevant_3c = [pos for pos in window_range if pos in codon_pos_three]

        window_results = []

        for category in df['TSI_category'].cat.categories:
            current_data = df[df['TSI_category'] == category]

            if relevant_12c:
                maf12c_mean = current_data.loc[:, relevant_12c].mean(axis=1, skipna = True).mean(skipna = True)
            else:
                maf12c_mean = np.nan

            if relevant_3c:
                maf3c_mean = current_data.loc[:, relevant_3c].mean(axis=1,  skipna = True).mean(skipna = True)
            else:
                maf3c_mean = np.nan

            for tsi_day in current_data['TSI_days'].unique():
                temp_df = pd.DataFrame({
                        'Window_Start': [start],
                        'Window_End': [end],
                        'Window_Centre': [centre],
                        'MAF12c_Mean': [maf12c_mean],
                        'MAF3c_Mean': [maf3c_mean],
                        'TSI_category': [category],
                        'TSI_days': [tsi_day]
                    })

                window_results.append(temp_df)

        if window_results:
            window_df = pd.concat(window_results, ignore_index=True)
            results_list.append(window_df)

    results_df = pd.concat(results_list, ignore_index=True)

    #SAVE TABLE AS CSV 
    results_df.to_csv(output_file, index=False)