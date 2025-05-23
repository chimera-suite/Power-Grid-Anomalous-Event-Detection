import pandas as pd
import matplotlib.pyplot as plt
from capymoa.stream import ARFFStream
from capymoa.evaluation import prequential_evaluation
from capymoa.classifier import NaiveBayes, HoeffdingTree, EFDT, KNN, PassiveAggressiveClassifier, SGDClassifier, HoeffdingAdaptiveTree, SAMkNN, CSMOTE, WeightedkNN
from capymoa.classifier import OnlineBagging, OnlineAdwinBagging, LeveragingBagging, OzaBoost, OnlineSmoothBoost, StreamingRandomPatches, StreamingGradientBoostedTrees, AdaptiveRandomForestClassifier
from scipy.io import arff
import tempfile
import numpy as np
import os
import random
import glob
from typing import List

SEEDS = list(range(30))

def extract_embedding_features(arff_path: str) -> List[str]:
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    return [col for col in df.columns if col.startswith("Embedding_Feature_")]


stream_features = [
    "Current_[A]", "Battery_Temperature_[°C]", "Ambient_Temperature_[°C]"
]
ke_features = [
    "KE_cntUp", "KE_pTot", "KE_qTot", "loadGroups", "serviceLocations"
]
always_used = ["Year", "Month", "Day", "Hour", "Substation_ID"]
embedding_features = [f"Embedding_Feature_{i}" for i in range(8)]
class_attr = "Target"

"""
scenarios_full = {
    "Baseline_Stream": always_used + stream_features,
    "Baseline_KE": always_used + ke_features,
    "Baseline_EMB": always_used + embedding_features,
    "Scenario_STREAM+KE": always_used + stream_features + ke_features,
    "Scenario_STREAM+EMB": always_used + stream_features + embedding_features,
    "Scenario_ALL": always_used + stream_features + ke_features + embedding_features,
}"""



# === Creating ARFF file ===
def create_temp_arff(original_path: str, selected_columns: List[str]):
    data, meta = arff.loadarff(original_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    df = df[selected_columns + [class_attr]]

    nominal_values = {
        "Substation_ID": sorted(df["Substation_ID"].unique().tolist())
    }

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".arff", mode="w", encoding="utf-8")
    tmp_file.write("@RELATION filtered_stream\n\n")

    for col in df.columns:
        if col == "Substation_ID":
            values = ",".join(sorted(df[col].unique()))
            tmp_file.write(f"@ATTRIBUTE {col} {{{values}}}\n")
        elif col == "Target":
            tmp_file.write("@ATTRIBUTE Target {0,1}\n")
        elif df[col].dtype == object:
            tmp_file.write(f"@ATTRIBUTE {col} STRING\n")
        elif df[col].dtype.kind in ['i', 'f']:
            tmp_file.write(f"@ATTRIBUTE {col} NUMERIC\n")
        else:
            tmp_file.write(f"@ATTRIBUTE {col} STRING\n")


    tmp_file.write("\n@DATA\n")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    df.to_csv(tmp_file, index=False, header=False)
    tmp_file.close()
    return tmp_file.name

def run_scenario(name, selected_features, file_path):

    base_dir = os.path.splitext(os.path.basename(file_path))[0]

    temp_path = create_temp_arff(file_path, selected_features)

    cumulative_metrics_all = {}
    windowed_results_all = []

    for run_id, seed in enumerate(SEEDS):
        np.random.seed(seed)
        random.seed(seed)

        temp_path = create_temp_arff(file_path, selected_features)

        stream = ARFFStream(path=temp_path, class_index=-1)
        stream.restart()

        model = LeveragingBagging(base_learner=EFDT(schema = stream.get_schema()), random_seed=seed, schema=stream.get_schema())  # Cambia modello qui se vuoi

        # Prequential Evaluation
        results = prequential_evaluation(stream, model, window_size=1200, store_predictions=True, store_y=True)
        os.remove(temp_path)

        cumulative = results.cumulative.metrics_dict()
        for key, value in cumulative.items():
            cumulative_metrics_all.setdefault(key, []).append(value)

        df = pd.DataFrame(results.metrics_per_window())
        windowed_results_all.append(df)
    
    wanted_metrics = [
        "accuracy", "f1_score", "kappa", 
        "precision", "precision_0", "precision_1", 
        "recall", "recall_0", "recall_1"
    ]

    filtered_cumulative = {
        metric: cumulative_metrics_all[metric]
        for metric in wanted_metrics if metric in cumulative_metrics_all
    }

    cumulative_df = pd.DataFrame(filtered_cumulative).T
    cumulative_df.columns = [f"Run_{i+1}" for i in range(len(SEEDS))]
    cumulative_df = cumulative_df.round(2)
    cumulative_df.to_csv("cumualtive.csv")

    metrici_da_tenere = ["accuracy", "precision", "recall", "kappa", "f1_score"]
    all_metrics = {metric: [] for metric in metrici_da_tenere}

    for df in windowed_results_all:
        for metric in metrici_da_tenere:
            all_metrics[metric].append(df[metric].values)

    media_varianza = {
        metric: {
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0)
        } for metric, values in all_metrics.items()
    }

    window_count = len(media_varianza["accuracy"]["mean"])
    summary_df = pd.DataFrame({
        "Window": range(window_count),
        **{f"{metric}_mean": media_varianza[metric]["mean"] for metric in metrici_da_tenere},
        **{f"{metric}_std": media_varianza[metric]["std"] for metric in metrici_da_tenere}
    })
    summary_df = summary_df.round(2)
    summary_df = summary_df.fillna(0.0)
    summary_df.to_csv("windowed.csv")

    # === PLOT ===
    plt.figure(figsize=(10, 6))
    for metric in metrici_da_tenere:
        plt.plot(media_varianza[metric]["mean"], label=f"{metric} (mean)")

    plt.xlabel("Window Index")
    plt.ylabel("Metric Value (%)")
    plt.title(f"Windowed Mean Metrics - {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot.png")

stream_paths = sorted(glob.glob("")) #stream path

for file_path in stream_paths:
    embedding_features = extract_embedding_features(file_path)

    scenarios_emb = {
    "Baseline_EMB": always_used + embedding_features,
    "Scenario_STREAM+KE": always_used + stream_features + ke_features,
    "Scenario_STREAM+EMB": always_used + stream_features + embedding_features,
    "Scenario_ALL": always_used + stream_features + ke_features + embedding_features,
    }   

    for scenario_name, features in scenarios_emb.items():
        run_scenario(scenario_name, features, file_path)