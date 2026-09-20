import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


load_dotenv('../.env')
load_dotenv('.env')


def main():
    parser = ArgumentParser(
        description='Count NaN values in a training feature split.'
    )
    parser.add_argument(
        '--endpoint',
        choices=['os', 'pfs'],
        required=True,
        help='Survival endpoint.'
    )
    parser.add_argument(
        '--shuffle',
        type=int,
        choices=range(10),
        required=True,
        help='Split shuffle identifier (0-9).'
    )
    parser.add_argument(
        '--fold',
        type=int,
        choices=range(5),
        required=True,
        help='Split fold identifier (0-4).'
    )
    args = parser.parse_args()

    split_root = Path(os.environ.get('SPLITDATADIR', 'data/splits'))
    feature_file = (
        split_root
        / str(args.shuffle)
        / str(args.fold)
        / f'train_features_{args.endpoint}_processed_nan.parquet'
    )
    if not feature_file.exists():
        parser.error(f'training feature file does not exist: {feature_file}')

    features = pd.read_parquet(feature_file)
    all_count = int(features.shape[0]*features.shape[1])
    print(all_count)
    nan_count = int(features.isna().sum().sum())
    print(nan_count)
    print(f"%Missing: {nan_count/all_count*100:.2f}")


if __name__ == '__main__':
    main()
