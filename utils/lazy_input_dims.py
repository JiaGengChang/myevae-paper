import pandas as pd 

def lazy_input_dims(df:pd.DataFrame, params:dict):
    # update parameters dictionary with correct input dimensions
    # df = training dataframe
    # params - current parameters with unknown input_dims
    # find column by regex based on input abbrv
    find_column = {'cth' : 'Feature_chromothripsis',
                   'apobec': 'Feature_APOBEC',
                   'clin': 'Feature_clin',
                   'exp': 'Feature_exp',
                   'sbs': 'Feature_SBS',
                   'ig': 'Feature_(RNASeq|SeqWGS)',
                   'gistic': 'Feature_CNA_(Amp|Del)',
                   'fish': 'Feature_fish',
                   'cna': 'Feature_CNA_ENSG'}
    # lazy determination of input dimensions
    params.input_dims = [
        params.input_dims[i] 
        if params.input_dims[i] 
        else df.filter(regex=find_column[params.input_types[i]]).columns.__len__() 
        for i in range(len(params.input_types))
    ]
    return params
