####### PRE-PROCESSING HELPER FUNCTIONS
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split


# Load data
def load_data(
    dataset, subtypes, target_name, dataset_sheet_name=None, subtype_sheet_name=None
):
    if subtypes.endswith(".xlsx"):
        subtypes = pd.read_excel(
            subtypes, sheet_name=subtype_sheet_name, header=1, index_col=0
        )
    else:
        subtypes = pd.read_csv(subtypes, sep="\t", index_col=0)

    if dataset.endswith(".xlsx"):
        df = pd.read_excel(dataset, sheet_name=dataset_sheet_name)
    else:
        df = pd.read_csv(dataset, sep="\t")

    df = df.transpose()

    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header  # set the header row as the df header

    data = pd.concat([df, subtypes[[target_name]]], axis=1, join="inner")

    data.columns = data.columns.str.strip()

    return data


# Split into X and y
def get_training(data, target_name):
    y = data.filter(regex=f"^{target_name}")
    # Get columns to drop
    columns_to_drop = [col for col in data.columns if col.startswith(target_name)]

    # Drop the columns
    X = data.drop(columns_to_drop, axis=1)
    X, y = correct_dtypes(X, y)

    return X, y


from functools import reduce


from functools import reduce


# Load all datasets and resize
def load_resize(datasets, labels, target_name):
    data = {}

    for modality in datasets:
        data[modality] = load_data(datasets[modality], labels, target_name)

    suffixed = {}
    target = {}

    # Add suffixes to each DataFrame
    for key in data.keys():
        target[key] = data[key].filter(regex=f"^{target_name}")

        # Get the columns excluding the target column
        columns_to_suffix = data[key].drop(columns=target[key].columns)

        # Add suffix only to non-target columns
        suffixed[key] = columns_to_suffix.add_suffix("_" + key)
        # suffixed[key] = pd.concat([target, suffixed[key]], axis=1, join='inner')

    # Getting first column as target, since they are now all the same
    y = pd.concat(target.values(), axis=1, join="inner").iloc[:, 0]

    # Merge datasets based on index and aligning them with the target
    merged = pd.concat(
        [pd.concat(suffixed.values(), axis=1, join="inner"), y], axis=1, join="inner"
    )

    merged_X = (merged.drop(columns=y.name)).astype("float")

    # Encoding the variables
    enc = LabelEncoder()
    y = enc.fit_transform(y)

    X = {}

    for key in data.keys():
        # Extract the corresponding columns for each dataset
        X[key] = (merged.loc[:, suffixed[key].columns]).astype("float")

    return merged_X, X, y


def train_test_val_split(data, target):
    x, eval_data, y, eval_target = train_test_split(
        data, target, test_size=0.2, train_size=0.8
    )
    train_data, val_data, train_target, val_target = train_test_split(
        x, y, test_size=0.25, train_size=0.75
    )

    return train_data, val_data, eval_data, train_target, val_target, eval_target


def correct_dtypes(data, target):
    enc = LabelEncoder()
    target = enc.fit_transform(target)
    # target = target.astype("category")

    data = data.astype("float")

    return data, target
