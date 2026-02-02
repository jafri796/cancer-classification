"""
WORKFLOW 5: Setup MLflow Tracking
Initialize experiment tracking infrastructure
"""

import sys
import subprocess
import time
from pathlib import Path

print("\n" + "="*80)
print("WORKFLOW 5: MLFLOW TRACKING SETUP")
print("="*80)

sys.path.insert(0, '.')

try:
    import mlflow
    print("\n✓ MLflow imported successfully")
    
    # Create mlruns directory if needed
    Path("mlruns").mkdir(exist_ok=True)
    
    print("\n📌 Configuring MLflow tracking...")
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    print("  ✓ Tracking URI set to: file:./mlruns")
    
    # Create experiment
    experiment_name = "pcam_baseline"
    try:
        exp_id = mlflow.create_experiment(experiment_name)
        print(f"  ✓ Experiment created: '{experiment_name}' (ID: {exp_id})")
    except:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp:
            print(f"  ✓ Experiment exists: '{experiment_name}' (ID: {exp.experiment_id})")
    
    # Log a test run
    print("\n📌 Logging test experiment run...")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="test_baseline"):
        mlflow.log_param("model", "resnet50")
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("learning_rate", 0.001)
        
        mlflow.log_metric("accuracy", 0.92, step=0)
        mlflow.log_metric("sensitivity", 0.94, step=0)
        mlflow.log_metric("specificity", 0.91, step=0)
        mlflow.log_metric("auc", 0.965, step=0)
        
        print("  ✓ Logged parameters: model, batch_size, learning_rate")
        print("  ✓ Logged metrics: accuracy, sensitivity, specificity, auc")
        
        # Create a test artifact
        with open("test_artifact.txt", "w") as f:
            f.write("MLflow integration test\nPCam baseline model setup")
        
        mlflow.log_artifact("test_artifact.txt")
        print("  ✓ Logged artifact: test_artifact.txt")
    
    print("\n✓ WORKFLOW 5 COMPLETE: MLflow tracking configured")
    print("\n📌 To view MLflow dashboard:")
    print("  Run: mlflow ui --host 0.0.0.0 --port 5000")
    print("  Then open: http://localhost:5000")
    
except Exception as e:
    print(f"\n❌ Error in workflow 5: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✓ WORKFLOW 5 COMPLETE")
print("="*80)
