import pandas as pd

def generate_lrtt_features(df, gag, pol, gp120, gp41, feature = 'normalised.largest.rtt'):
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
        genome_dual = group['solo.dual.count'].mean()
        genome_reads = group['reads'].mean()
        
        # Gene-level averages
        gag_lrtt = group[group['xcoord'].isin(gag)][feature].mean()
        pol_lrtt = group[group['xcoord'].isin(pol)][feature].mean()
        gp120_lrtt = group[group['xcoord'].isin(gp120)][feature].mean()
        gp41_lrtt = group[group['xcoord'].isin(gp41)][feature].mean()
        
        gag_tips = group[group['xcoord'].isin(gag)]['tips'].mean()
        pol_tips = group[group['xcoord'].isin(pol)]['tips'].mean()
        gp120_tips = group[group['xcoord'].isin(gp120)]['tips'].mean()
        gp41_tips = group[group['xcoord'].isin(gp41)]['tips'].mean()
        
        gag_dual = group[group['xcoord'].isin(gag)]['solo.dual.count'].mean()
        pol_dual = group[group['xcoord'].isin(pol)]['solo.dual.count'].mean()
        gp120_dual = group[group['xcoord'].isin(gp120)]['solo.dual.count'].mean()
        gp41_dual = group[group['xcoord'].isin(gp41)]['solo.dual.count'].mean()

        gag_reads = group[group['xcoord'].isin(gag)]['reads'].mean()
        pol_reads = group[group['xcoord'].isin(pol)]['reads'].mean()
        gp120_reads = group[group['xcoord'].isin(gp120)]['reads'].mean()
        gp41_reads = group[group['xcoord'].isin(gp41)]['reads'].mean()

        #save tsi
        tsi_days = group['TSI_days'].iloc[0]

        results.append({
            'RENAME_ID': name,
            'TSI_days': tsi_days,
            'genome_lrtt': genome_lrtt,
            'genome_tips': genome_tips,
            'genome_dual': genome_dual,
            'genome_reads': genome_reads,
            'gag_lrtt': gag_lrtt,
            'gag_tips': gag_tips,
            'gag_dual': gag_dual,
            'gag_reads': gag_reads,
            'pol_lrtt': pol_lrtt,
            'pol_tips': pol_tips,
            'pol_dual': pol_dual,
            'pol_reads': pol_reads,
            'gp120_lrtt': gp120_lrtt,
            'gp120_tips': gp120_tips,
            'gp120_dual': gp120_dual,
            'gp120_reads': gp120_reads,
            'gp41_lrtt': gp41_lrtt,
            'gp41_tips': gp41_tips,
            'gp41_dual': gp41_dual,
            'gp41_reads': gp41_reads
        })
    
    return pd.DataFrame(results)

def generate_maf_features(df, gag, pol, gp120, gp41, feature_12c = 'MAF12c_Mean', feature_3c = 'MAF3c_Mean'):
    ''' Generate aggregated LRTT predictors. '''
    results = []
    
    # Group by RENAME_ID
    grouped = df.groupby('RENAME_ID')
    
    
    for name, group in grouped:
        
        # Convert xcoord to integer for matching
        group['Window_Centre'] = group['Window_Centre'].astype(int)
        group['Window_Centre'] = group['Window_Centre'].astype(int)
        # Genome-level average
        genome_maf12c = group[feature_12c].mean()
        genome_maf3c = group[feature_3c].mean()
        
        # Gene-level averages
        gag_maf12c = group[group['Window_Centre'].isin(gag)][feature_12c].mean()
        pol_maf12c = group[group['Window_Centre'].isin(pol)][feature_12c].mean()
        gp120_maf12c = group[group['Window_Centre'].isin(gp120)][feature_12c].mean()
        gp41_maf12c = group[group['Window_Centre'].isin(gp41)][feature_12c].mean()

        # Gene-level averages
        gag_maf3c = group[group['Window_Centre'].isin(gag)][feature_3c].mean()
        pol_maf3c = group[group['Window_Centre'].isin(pol)][feature_3c].mean()
        gp120_maf3c = group[group['Window_Centre'].isin(gp120)][feature_3c].mean()
        gp41_maf3c = group[group['Window_Centre'].isin(gp41)][feature_3c].mean()
        
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

def generate_GC_features(df, gag, pol, gp120, gp41, feature = 'window_gc_ratio'):
    ''' Generate aggregated LRTT predictors. '''
    results = []
    
    # Group by RENAME_ID
    grouped = df.groupby('RENAME_ID')
    
    for name, group in grouped:
        # Convert xcoord to integer for matching
        group['window_centre'] = group['window_centre'].astype(int)
        group['window_centre'] = group['window_centre'].astype(int)
        # Genome-level average
        genome_GCratio = group[feature].mean()
        
        # Gene-level averages
        gag_GCratio = group[group['window_centre'].isin(gag)][feature].mean()
        pol_GCratio = group[group['window_centre'].isin(pol)][feature].mean()
        gp120_GCratio = group[group['window_centre'].isin(gp120)][feature].mean()
        gp41_GCratio = group[group['window_centre'].isin(gp41)][feature].mean()
        
        #save tsi
        tsi_days = group['TSI_days'].iloc[0]

        results.append({
            'RENAME_ID': name,
            'TSI_days': tsi_days,
            'genome_GCratio': genome_GCratio,
            'gag_GCratio': gag_GCratio,
            'pol_GCratio': pol_GCratio,
            'gp120_GCratio': gp120_GCratio,
            'gp41_GCratio': gp41_GCratio
        })
    
    return pd.DataFrame(results)