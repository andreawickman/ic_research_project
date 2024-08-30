#Function to rename feature sets of final analyses
def abbreviate_feature_sets(df, column='Feature_Set'):
    abbreviations = {
        'base_plus_genome_ambig': 'Base + genome_ambig',
        'base_plus_pol_ambig': 'Base + pol_ambig',
        'base_plus_gp120_ambig': 'Base + gp120_ambig',
        'base_plus_lrtt_coeff_1': 'Base + lrtt_coeff_1', 
        'base_plus_lrtt_coeff_11': 'Base + lrtt_coeff_11', 
        'base_plus_genome_ambig_gp120_ambig': 'Base + genome_gp120_ambig',
        'base_plus_genome_ambig_pol_ambig': 'Base + genome_pol_ambig',
        'base_plus_genome_ambig_lrtt_coeff_1': 'Base + genome_ambig + lrtt_coeff_1',
        'base_plus_genome_ambig_lrtt_coeff_11': 'Base + genome_ambig + lrtt_coeff_11',
        'base_plus_pol_ambig_gp120_ambig': 'Base + pol_gp120_ambig',
        'base_plus_pol_ambig_lrtt_coeff_1': 'Base + pol_ambig + lrtt_coeff_1',
        'base_plus_pol_ambig_lrtt_coeff_11': 'Base + pol_ambig + lrtt_coeff_11',
        'base_plus_gp120_ambig_lrtt_coeff_1': 'Base + gp120_ambig + lrtt_coeff_1',
        'base_plus_gp120_ambig_lrtt_coeff_11': 'Base + gp120_ambig + lrtt_coeff_11',
        'base_plus_lrtt_coeff_1_lrtt_coeff_11': 'Base + All Coeffs',
        'base_plus_genome_ambig_pol_ambig_gp120_ambig': 'Base + All Ambig',
        'base_plus_genome_ambig_pol_ambig_lrtt_coeff_1': 'Base + genome_pol_ambig + lrtt_coeff_1',
        'base_plus_genome_ambig_pol_ambig_lrtt_coeff_11': 'Base + genome_pol_ambig + lrtt_coeff_11',
        'base_plus_genome_ambig_gp120_ambig_lrtt_coeff_1': 'Base + genome_gp120_ambig + lrtt_coeff_1',
        'base_plus_genome_ambig_gp120_ambig_lrtt_coeff_11': 'Base + genome_gp120_ambig + lrtt_coeff_11',
        'base_plus_genome_ambig_lrtt_coeff_1_lrtt_coeff_11': 'Base + genome_ambig + All Coeffs',
        'base_plus_pol_ambig_gp120_ambig_lrtt_coeff_1': 'Base + pol_gp120_ambig + lrtt_coeff_1',
        'base_plus_pol_ambig_gp120_ambig_lrtt_coeff_11': 'Base + pol_gp120_ambig + lrtt_coeff_11',
        'base_plus_pol_ambig_lrtt_coeff_1_lrtt_coeff_11': 'Base + pol_ambig + All Coeffs',
        'base_plus_gp120_ambig_lrtt_coeff_1_lrtt_coeff_11': 'Base + gp120_ambig + All Coeffs',
        'base_plus_genome_ambig_pol_ambig_gp120_ambig_lrtt_coeff_1': 'Base + All Ambig + lrtt_coeff_1',
        'base_plus_genome_ambig_pol_ambig_gp120_ambig_lrtt_coeff_11': 'Base + All Ambig + lrtt_coeff_11',
        'base_plus_genome_ambig_pol_ambig_lrtt_coeff_1_lrtt_coeff_11': 'Base + genome_pol_ambig + All Coeffs',
        'base_plus_genome_ambig_gp120_ambig_lrtt_coeff_1_lrtt_coeff_11': 'Base + genome_gp120_ambig + All Coeffs',
        'base_plus_pol_ambig_gp120_ambig_lrtt_coeff_1_lrtt_coeff_11': 'Base + pol_gp120_ambig + All Coeffs',
        'base_plus_genome_ambig_pol_ambig_gp120_ambig_lrtt_coeff_1_lrtt_coeff_11':'Base + All Ambig + All Coeffs',
    }
    
    #apply clarified feature set names
    df[column] = df[column].replace(abbreviations)
    return df