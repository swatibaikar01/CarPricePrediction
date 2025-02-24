import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="Path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, help="Path to save train data")
    parser.add_argument("--test_data", type=str, help="Path to save test data")
    args = parser.parse_args()

    # Start MLflow Run
    mlflow.start_run()

    # Log arguments
    logging.info(f"Input data path: {args.raw_data}")
    logging.info(f"Test-train ratio: {args.test_train_ratio}")

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Encode categorical feature
    # Ensure 'Segment' column exists before encoding
    if 'Segment' in df.columns:
        le = LabelEncoder()
        df['Segment'] = le.fit_transform(df['Segment'].astype(str))  
    else:
        logging.warning("'Segment' column not found, skipping encoding.")

    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)  #  Write code to split the data into train and test datasets

    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)  # Create directories for train_data and test_data
    os.makedirs(args.test_data, exist_ok=True)  # Create directories for train_data and test_data
    train_df.to_csv(os.path.join(args.train_data, "train_data.csv"), index=False)  # Specify the name of the train data file
    test_df.to_csv(os.path.join(args.test_data, "test_data.csv"), index=False)  # Specify the name of the test data file

    # log the metrics
    mlflow.log_metric('train size', train_df.shape[0])  # Log the train dataset size
    mlflow.log_metric('test size', test_df.shape[0])  # Log the test dataset size
    
    mlflow.end_run()

if __name__ == "__main__":
    main()
