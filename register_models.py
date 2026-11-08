import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv(dotenv_path=".env", override=True)
c = MlflowClient()
exp = c.get_experiment_by_name("visionguard")
NAME = "visionguard"
versions = {}

for run_name in ["img_size_160", "efficientnetb0"]:
    run = c.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")[0]
    lm = mlflow.search_logged_models(experiment_ids=[exp.experiment_id],
                                     filter_string=f"source_run_id = '{run.info.run_id}'", output_format="list")[0]
    mv = mlflow.register_model(f"models:/{lm.model_id}", NAME, tags={
        "run_name": run_name, "threshold": f"{run.data.metrics['threshold']:.3f}",
        "test_roc_auc": f"{run.data.metrics['test_roc_auc']:.4f}", "git_commit": run.data.tags["git_commit"]})
    versions[run_name] = mv.version
    print("registered", run_name, "as version", mv.version)

c.set_registered_model_alias(NAME, "champion", versions["efficientnetb0"])
c.set_registered_model_alias(NAME, "challenger", versions["img_size_160"])
print("champion ->", versions["efficientnetb0"], "| challenger ->", versions["img_size_160"])
