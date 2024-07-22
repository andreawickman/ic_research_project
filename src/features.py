import pandas as pd 

def generate_lrtt_features(df, gag, pol, gp120, gp41):
    ''' Generate aggregated LRTT predictors. '''
    results = []
    
    # Group by RENAME_ID
    grouped = df.groupby('RENAME_ID')
    
    for name, group in grouped:
        # Convert xcoord to integer for matching
        group['xcoord'] = group['xcoord'].astype(int)
        # Genome-level average
        genome_lrtt = group['normalised.largest.rtt'].mean()
        genome_tips = group['tips'].mean()
        
        # Gene-level averages
        gag_lrtt = group[group['xcoord'].isin(gag)]['normalised.largest.rtt'].mean()
        pol_lrtt = group[group['xcoord'].isin(pol)]['normalised.largest.rtt'].mean()
        gp120_lrtt = group[group['xcoord'].isin(gp120)]['normalised.largest.rtt'].mean()
        gp41_lrtt = group[group['xcoord'].isin(gp41)]['normalised.largest.rtt'].mean()
        
        gag_tips = group[group['xcoord'].isin(gag)]['tips'].mean()
        pol_tips = group[group['xcoord'].isin(gag)]['tips'].mean()
        gp120_tips = group[group['xcoord'].isin(gag)]['tips'].mean()
        gp41_tips = group[group['xcoord'].isin(gag)]['tips'].mean()

        #save tsi
        tsi_days = group['TSI_days'].iloc[0]

        results.append({
            'RENAME_ID': name,
            'TSI_days': tsi_days,
            'genome_lrtt': genome_lrtt,
            'genome_tips': genome_tips,
            'gag_lrtt': gag_lrtt,
            'gag_tips': gag_tips,
            'pol_lrtt': pol_lrtt,
            'pol_tips': pol_tips,
            'gp120_lrtt': gp120_lrtt,
            'gp120_tips': gp120_tips,
            'gp41_lrtt': gp41_lrtt,
            'gp41_tips': gp41_tips
        })
    
    return pd.DataFrame(results)

def generate_maf_features(df, gag, pol, gp120, gp41):
    ''' Generate aggregated LRTT predictors. '''
    results = []
    
    # Group by RENAME_ID
    grouped = df.groupby('RENAME_ID')
    
    
    for name, group in grouped:
        
        # Convert xcoord to integer for matching
        group['Window_Centre'] = group['Window_Centre'].astype(int)
        group['Window_Centre'] = group['Window_Centre'].astype(int)
        # Genome-level average
        genome_maf12c = group['MAF12c_Mean'].mean()
        genome_maf3c = group['MAF3c_Mean'].mean()
        
        # Gene-level averages
        gag_maf12c = group[group['Window_Centre'].isin(gag)]['MAF12c_Mean'].mean()
        pol_maf12c = group[group['Window_Centre'].isin(pol)]['MAF12c_Mean'].mean()
        gp120_maf12c = group[group['Window_Centre'].isin(gp120)]['MAF12c_Mean'].mean()
        gp41_maf12c = group[group['Window_Centre'].isin(gp41)]['MAF12c_Mean'].mean()

        # Gene-level averages
        gag_maf3c = group[group['Window_Centre'].isin(gag)]['MAF3c_Mean'].mean()
        pol_maf3c = group[group['Window_Centre'].isin(pol)]['MAF3c_Mean'].mean()
        gp120_maf3c = group[group['Window_Centre'].isin(gp120)]['MAF3c_Mean'].mean()
        gp41_maf3c = group[group['Window_Centre'].isin(gp41)]['MAF3c_Mean'].mean()
        
        #save tsi
        tsi_days = group['TSI_days'].iloc[0]

        results.append({
            'RENAME_ID': name,
            'TSI_days': tsi_days,
            'genome_maf12c': genome_maf12c,
            'genome_maf3c': genome_maf3c,
            'gag_maf12c': gag_maf12c,
            'gag_maf3c': gag_maf3c,
            'pol_maf12c': pol_maf12c,
            'pol_maf3c': pol_maf3c,
            'gp120_maf12c': gp120_maf12c,
            'gp120_maf3c': gp120_maf3c,
            'gp41_maf12c': gp41_maf12c,
            'gp41_maf3c': gp41_maf3c,
        })
    
    return pd.DataFrame(results)