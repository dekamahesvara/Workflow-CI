import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def load_data(input_dir):
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="namadataset_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(args.input)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        model.score(X_test, y_test)


if __name__ == "__main__":
    main()
