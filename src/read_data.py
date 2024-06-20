import pyreadr
import pandas as pd
import numpy as np

def read_datasets(): 
    """
    Reads the specified RDS and CSV files and returns the data as pandas DataFrames.
    
    Returns:
        dict: A dictionary containing the DataFrames with keys as the file descriptors.
    """ 
    file_paths = {
        'agg_data': './data/raw/BW_tshedimoso_TSI.RDS',
        'phylo_features': './data/raw/BW_tshedimoso_phylo_patstats.RDS',
        'maf_data': './data/raw/Cleaned_BW_tshedimoso_MAFs.RDS',
        'rakai_maf_data': './data/raw/Rakai_MAFs.RDS',
        'rakai_phylo_data': './data/raw/Rakai_phylo_patstats.csv'
    }

    #READ RDS FILES
    agg_data = pyreadr.read_r(file_paths['agg_data'])[None]
    phylo_features = pyreadr.read_r(file_paths['phylo_features'])[None]
    maf_data = pyreadr.read_r(file_paths['maf_data'])[None]
    rakai_maf_data = pyreadr.read_r(file_paths['rakai_maf_data'])[None]
    
    #READ CSV FILE
    rakai_phylo_data = pd.read_csv(file_paths['rakai_phylo_data'])

    #ACCESS DATAFRAMES
    data_frames = {
        'agg_data_df': agg_data,
        'phylo_features_df': phylo_features,
        'maf_data_df': maf_data,
        'rakai_maf_data_df': rakai_maf_data,
        'rakai_phylo_data_df': rakai_phylo_data
    }
    return data_frames


def merge_phylo_data(botswana_data, rakai_data):

    common_columns = ['TSI_days', 'tree.id', 'xcoord', 'tips', 'reads', 'subgraphs', 'clades',
                  'overall.rtt', 'largest.rtt', 'max.branch.length', 'max.pat.distance',
                  'global.mean.pat.distance', 'normalised.largest.rtt', 'normalised.max.branch.length',
                  'normalised.max.pat.distance', 'normalised.global.mean.pat.distance',
                  'normalised.subgraph.mean.pat.distance', 'solo.dual.count']

    botswana_data = botswana_data[['pt_letter'] + common_columns]
    rakai_data = rakai_data[['host.id', 'AID'] + common_columns]
    #round up xcoord for consistency
    rakai_data['xcoord'] = np.ceil(rakai_data['xcoord'])

    botswana_data = botswana_data.rename(columns={'pt_letter': 'host.id'})
    botswana_data.loc[:, 'host.id'] = botswana_data['host.id'].astype(str)
    rakai_data.loc[:, 'host.id'] = rakai_data['host.id'].astype(str)

    columns_to_convert_to_float = ['TSI_days', 'xcoord', 'tips', 'reads', 'subgraphs', 'clades', 'solo.dual.count']

    for col in columns_to_convert_to_float:
        botswana_data.loc[:, col] = botswana_data[col].astype(float)
        rakai_data.loc[:, col] = rakai_data[col].astype(float)

    merged_df = pd.merge(botswana_data, rakai_data, on=['host.id'] + common_columns, how='outer')
    #SAVE TABLE AS CSV 
    merged_df.to_csv('./data/derived/merged_phylo_data.csv', index=False)

    return merged_df


##TANYAS FUNCTION (https://github.com/BDI-pathogens/HIV-phyloTSI/blob/main/HIVPhyloTSI.py)
def load_reference_data(modeldir):
    ''' Load HXB2 data.'''
    #  HXB2 positions
    hxb2 = pd.read_csv('{}/HXB2_refdata.csv'.format(modeldir))
    hxb2['position']=hxb2['HXB2 base position']
    hxb2.set_index('position' , inplace=True)
    # Third codon position sites
    rf3_3cp = hxb2.groupby(['RF3 protein', 'RF3 aa position'])['HXB2 base position'].max()
    rf2_3cp = hxb2.groupby(['RF2 protein', 'RF2 aa position'])['HXB2 base position'].max()
    rf1_3cp = hxb2.groupby(['RF1 protein', 'RF1 aa position'])['HXB2 base position'].max()
    # Make into set
    t1 = set(rf1_3cp.reset_index()['HXB2 base position'])
    t2 = set(rf2_3cp.reset_index()['HXB2 base position'])
    t3 = set(rf3_3cp.reset_index()['HXB2 base position'])
    # Same for 1st/2nd position
    rf1_12 = set(hxb2.groupby(['RF1 protein', 'RF1 aa position'])['HXB2 base position'].nsmallest(2).reset_index()['HXB2 base position'])
    rf2_12 = set(hxb2.groupby(['RF2 protein', 'RF2 aa position'])['HXB2 base position'].nsmallest(2).reset_index()['HXB2 base position'])
    rf3_12 = set(hxb2.groupby(['RF3 protein', 'RF3 aa position'])['HXB2 base position'].nsmallest(2).reset_index()['HXB2 base position'])
    # Summarise
    first_second_codon_pos = set(rf1_12 | rf2_12 | rf3_12)
    third_codon_pos = set(t1 | t2 | t3) - first_second_codon_pos

    return first_second_codon_pos, third_codon_pos