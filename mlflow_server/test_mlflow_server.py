import mlflow
import os
import sys
import socket
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()
server_uri = os.getenv("MLFLOW_SERVER_URI")

if not server_uri:
    print("Error: MLFLOW_SERVER_URI not found in the .env file.")
    sys.exit(1)

def check_server_connection(uri, timeout=3):
    """Parses a URI and checks if the host is reachable on its port."""
    print(f"Pinging MLflow server at {uri}...")
    try:
        parsed_uri = urlparse(uri)
        host = parsed_uri.hostname
        port = parsed_uri.port

        if not host or not port:
            print(f"Error: Invalid URI '{uri}'. Could not determine host or port.")
            return False

        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                print("server is reachable.")
                return True
    except (socket.error, TypeError) as ex:
        print(f"Socket error: {ex}")
    
    print(f"Error: Could not connect to MLflow server at {uri}.")
    print("   Please ensure the server is running and the URI is correct.")
    return False

if __name__ == "__main__":
    if not check_server_connection(server_uri):
        sys.exit(1)
    
    mlflow.set_tracking_uri(server_uri)                  # <<<<---------------------------- MLFLOW start here
    mlflow.set_experiment("test connection mlflow server")

    with mlflow.start_run(run_name="RandomForest_v3_parsed") as run:
        print(f"\nStarting run with ID: {run.info.run_id}")

        # Log Parameters
        params = {"n_estimators": 200, "max_depth": 7, "random_state": 42}
        mlflow.log_params(params)
        print("Logged parameters:", params)

        # Log a Metric
        mlflow.log_metric("rmse", 0.45)
        print("Logged metric: rmse = 0.45")

        # Log an Artifact
        with open("feature_importance.txt", "w") as f:
            f.write("feature_A: 0.6\nfeature_B: 0.3\nfeature_C: 0.1")
        mlflow.log_artifact("feature_importance.txt", artifact_path="analysis")
        print("Logged artifact: feature_importance.txt")
        os.remove("feature_importance.txt")

    print("\nRun completed successfully!")