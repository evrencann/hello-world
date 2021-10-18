import pandas as pd
import argparse
from typing import Dict
import os


def is_accurate(pred: int, start: int, end: int) -> bool:
    """Compute whether a prediction is accurate enough

    Args:
        pred (int): predicted time series step of the center of the anomaly
        start (int): true start of anomaly
        end (int): true end of anomaly

    Returns:
        bool: prediction is accurate
    """
    accepted_inaccuracy = 100  # slack for short anomalies
    length_anomaly = end - start + 1
    if (
        min(start - length_anomaly, start - accepted_inaccuracy)
        < pred
        < max(end + length_anomaly, end + accepted_inaccuracy)
    ):
        return True
    else:
        return False


def accuracy(prediction: Dict[int, int], datapath: str) -> float:
    """Compute accuracy of anomaly predictions.

    Args:
        prediction (Dict[int, int]): maps data_id to center of predicted anomaly
        datapath (str): path to file containing the true anomaly data

    Returns:
        float: fraction of anomalies predicted correctly
    """
    data = pd.read_csv(datapath, index_col="data_id")

    n_accurate = 0
    for data_id, pred in prediction.items():
        true_start = data.loc[data_id, "anomaly_start"]
        true_end = data.loc[data_id, "anomaly_end"]
        n_accurate += int(is_accurate(pred, true_start, true_end))

    return n_accurate / len(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred_path",
        type=str,
        help="Path to csv file with predictions. "
        "Every line must be of the form: data_id,prediction_idx.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to csv file with anomaly data.",
        default=os.path.join(os.path.dirname(__file__), "metadata.csv"),
    )
    args = parser.parse_args()

    preds = {}
    with open(args.pred_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    for line in lines:
        data_id, pred = map(int, line.split(","))
        preds[data_id] = pred

    acc = accuracy(preds, args.data_path)
    print(f"Accuracy = {acc:.3f}")
