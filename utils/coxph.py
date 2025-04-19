from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.pipeline import Pipeline


def create_baseline_model(gepfeaturename:str='Feature_Clin_only',use_clin:bool=True)->Pipeline:
    """
    @ input
    gepfeaturename is a pattern that matches EXACTLY ONE of the GEP risk score column names
    The user is responsible for ensuring the specificity of this pattern to pull out one column name
    Otherwise, multiple risk scores may end up being used as features
    if no matches, then no GEP risk score is used i.e. pure clinical model
    For example, Feature_ClinOnly does not match any columns. 
    @ output
    a scikit learn Pipeline estimator, with fit, transform, and score functions
    """
    _ss = StandardScaler().set_output(transform='pandas')

    if use_clin:
        _transform = ColumnTransformer([
            ('clin.age', _ss, make_column_selector(pattern='Feature_clin_D_PT_age')),
            ('clin.other', 'passthrough', make_column_selector(pattern='Feature_clin_(?!D_PT_age)')),
            ('gep', 'passthrough', make_column_selector(pattern=gepfeaturename))
            ], remainder='drop'
        ).set_output(transform='pandas')
    else:
        _transform = ColumnTransformer([
            ('gep','passthrough', make_column_selector(pattern=gepfeaturename))
        ], remainder='drop'
        ).set_output(transform='pandas')
    
    _imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
    
    _estimator = CoxPHSurvivalAnalysis()
    
    model = Pipeline([
        ('transform', _transform),
        ('impute', _imputer),
        ('estimator', _estimator)
    ])
    
    return model