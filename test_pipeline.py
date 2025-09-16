#!/usr/bin/env python3
"""Quick pipeline test with the actual Titanic data."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from titanic_ml.core.models import ModelConfig, PipelineConfig
from titanic_ml.pipeline.pipeline import TitanicMLPipeline

def test_pipeline():
    """Test the complete pipeline with actual data."""
    
    # Configuration
    pipeline_config = PipelineConfig(
        data_path=Path("data"),
        model_output_path=Path("test_models"),
        use_polars=True,
        enable_hyperparameter_tuning=False,  # Disable for quick test
        n_trials=10
    )
    
    # Single model configuration (using Random Forest - no external deps)
    model_config = ModelConfig(
        model_type="random_forest",
        cv_folds=3,  # Fewer folds for quick test
        hyperparameters={"n_estimators": 50, "max_depth": 5}  # Quick params
    )
    
    try:
        print("🚀 Starting Titanic ML Pipeline Test")
        print(f"Data path: {pipeline_config.data_path}")
        print(f"Output path: {pipeline_config.model_output_path}")
        
        # Initialize pipeline
        pipeline = TitanicMLPipeline(pipeline_config)
        
        # Run the complete pipeline
        results = pipeline.run_full_pipeline(
            model_configs=[model_config],
            save_models=True,
            create_submission=True
        )
        
        print("\n✅ Pipeline completed successfully!")
        print(f"Results keys: {list(results.keys())}")
        
        if 'single_model' in results:
            model_info = results['single_model']
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                print(f"\n📊 Model Performance:")
                print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"   Precision: {metrics.get('precision', 0):.4f}")
                print(f"   Recall: {metrics.get('recall', 0):.4f}")
                print(f"   F1 Score: {metrics.get('f1_score', 0):.4f}")
                print(f"   ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        
        if 'submission_path' in results:
            print(f"\n📝 Submission file created: {results['submission_path']}")
            
        print(f"\n💾 Models saved to: {pipeline_config.model_output_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)