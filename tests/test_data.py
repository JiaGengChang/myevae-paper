from pathlib import Path

import pandas as pd


DATA_SPLITS = Path(__file__).parents[1] / 'data' / 'splits'


def test_sbs_feature_means_are_standardized():
    threshold = 1e-3

    for shuffle in range(10):
        for fold in range(5):
            for endpoint in ['os','pfs']:
                features_path = DATA_SPLITS / str(shuffle) / str(fold) / f'train_features_{endpoint}_processed_mut_nan.parquet'
                dataframe = pd.read_parquet(features_path)
                sbs_means = dataframe.filter(regex='Feature_SBS').mean()
                assert (len(sbs_means) > 0)
                assert (sbs_means < threshold).all(), (
                    f'{features_path} has SBS feature means at or above {threshold}: '
                    f'{sbs_means[sbs_means >= threshold].to_dict()}'
                )
