from pipelines.train_pipeline import train_pipeline
from steps.clean_data import clean_df
from steps.eval import evaluate_model
from steps.ingest_data import ingest_df
from steps.train_model import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    training = train_pipeline(
        ingest_df(),
        clean_df(),
        train_model(),
        evaluate_model(),
    )

    training.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
