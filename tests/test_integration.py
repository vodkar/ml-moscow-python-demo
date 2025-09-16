"""Integration tests for the complete ML pipeline."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from titanic_ml.config import AppConfig
from titanic_ml.pipeline.train import TrainingPipeline


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.fixture
    def sample_train_data(self) -> str:
        """Create sample training data CSV content."""
        return """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,30,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas",female,14,1,0,237736,30.0708,,C"""
    
    @pytest.fixture
    def sample_test_data(self) -> str:
        """Create sample test data CSV content."""
        return """PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
893,3,"Wilkes, Mrs. James",female,47,1,0,363272,7,,S
894,2,"Myles, Mr. Thomas Francis",male,62,0,0,240276,9.6875,,Q
895,3,"Wirz, Mr. Albert",male,27,0,0,315154,8.6625,,S
896,3,"Hirvonen, Mrs. Alexander",female,22,1,1,3101298,12.2875,,S"""
    
    @pytest.fixture
    def temp_config(self, sample_train_data: str, sample_test_data: str) -> AppConfig:
        """Create temporary configuration with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create data files
            train_file = temp_path / "train.csv"
            test_file = temp_path / "test.csv"
            
            train_file.write_text(sample_train_data)
            test_file.write_text(sample_test_data)
            
            # Create config
            config = AppConfig()
            config.data.train_file = train_file
            config.data.test_file = test_file
            config.training.model_save_path = temp_path / "models"
            config.training.metrics_save_path = temp_path / "metrics"
            config.prediction.output_path = temp_path / "output" / "predictions.csv"
            
            # Override model config for faster testing
            config.model.xgb_n_estimators = 10  # Reduce for speed
            config.training.cross_validation_folds = 2  # Reduce for speed
            
            yield config
    
    def test_full_training_pipeline(self, temp_config: AppConfig) -> None:
        """Test the complete training pipeline."""
        pipeline = TrainingPipeline(temp_config)
        
        # Validate configuration
        validation_results = pipeline.validate_configuration()
        assert validation_results["is_valid"]
        
        # Run training (using a simple model for speed)
        temp_config.model.model_type = "logistic_regression"  # Faster than XGBoost
        results = pipeline.run_full_pipeline()
        
        # Check that all components completed
        assert "model_path" in results
        assert "preprocessor_path" in results
        assert "performance_report_path" in results
        assert "validation_metrics" in results
        assert "cross_validation" in results
        
        # Check that files were created
        assert Path(results["model_path"]).exists()
        assert Path(results["preprocessor_path"]).exists()
        assert Path(results["performance_report_path"]).exists()
        
        # Check metrics are reasonable
        metrics = results["validation_metrics"]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        
        # Check cross-validation results
        cv_results = results["cross_validation"]
        assert isinstance(cv_results["individual_scores"], list)
        assert len(cv_results["individual_scores"]) == temp_config.training.cross_validation_folds
    
    def test_train_and_predict_workflow(self, temp_config: AppConfig) -> None:
        """Test training followed by prediction."""
        from titanic_ml.pipeline.predict import PredictionPipeline
        
        # Step 1: Train model
        train_pipeline = TrainingPipeline(temp_config)
        temp_config.model.model_type = "logistic_regression"
        train_results = train_pipeline.run_full_pipeline()
        
        # Step 2: Update prediction config with trained artifacts
        temp_config.prediction.model_path = Path(train_results["model_path"])
        temp_config.prediction.preprocessor_path = Path(train_results["preprocessor_path"])
        
        # Step 3: Make predictions
        predict_pipeline = PredictionPipeline(temp_config)
        prediction_results = predict_pipeline.run_batch_prediction(
            input_file=temp_config.data.test_file,
            with_analysis=True
        )
        
        # Check prediction results
        assert "predictions_file" in prediction_results
        assert "metadata" in prediction_results
        assert "analysis" in prediction_results
        
        # Check that predictions file exists and has correct format
        predictions_file = Path(prediction_results["predictions_file"])
        assert predictions_file.exists()
        
        predictions_df = pl.read_csv(predictions_file)
        assert "PassengerId" in predictions_df.columns
        assert "Survived" in predictions_df.columns
        assert len(predictions_df) > 0
        
        # Check that all predictions are 0 or 1
        unique_predictions = predictions_df["Survived"].unique().sort()
        assert all(pred in [0, 1] for pred in unique_predictions)
    
    def test_single_passenger_prediction(self, temp_config: AppConfig) -> None:
        """Test single passenger prediction."""
        from titanic_ml.pipeline.predict import PredictionPipeline
        
        # First train a model
        train_pipeline = TrainingPipeline(temp_config)
        temp_config.model.model_type = "logistic_regression"
        train_results = train_pipeline.run_full_pipeline()
        
        # Update prediction config
        temp_config.prediction.model_path = Path(train_results["model_path"])
        temp_config.prediction.preprocessor_path = Path(train_results["preprocessor_path"])
        
        # Make single prediction
        predict_pipeline = PredictionPipeline(temp_config)
        
        passenger_data = {
            "PassengerId": 999,
            "Pclass": 1,
            "Name": "Test, Mr. Passenger",
            "Sex": "male",
            "Age": 30,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "TEST123",
            "Fare": 50.0,
            "Cabin": "",
            "Embarked": "S",
        }
        
        result = predict_pipeline.predict_single_sample(passenger_data)
        
        # Check result format
        assert "passenger_id" in result
        assert "prediction" in result
        assert "survival_prediction" in result
        assert result["prediction"] in [0, 1]
        assert result["survival_prediction"] in ["Survived", "Did not survive"]
    
    def test_model_evaluation(self, temp_config: AppConfig) -> None:
        """Test model evaluation against ground truth."""
        import tempfile
        from titanic_ml.pipeline.predict import PredictionPipeline
        
        # Create ground truth data
        ground_truth_data = """PassengerId,Survived
892,0
893,1
894,0
895,0
896,1"""
        
        # Train model
        train_pipeline = TrainingPipeline(temp_config)
        temp_config.model.model_type = "logistic_regression"
        train_results = train_pipeline.run_full_pipeline()
        
        # Make predictions
        temp_config.prediction.model_path = Path(train_results["model_path"])
        temp_config.prediction.preprocessor_path = Path(train_results["preprocessor_path"])
        
        predict_pipeline = PredictionPipeline(temp_config)
        prediction_results = predict_pipeline.run_batch_prediction(
            input_file=temp_config.data.test_file,
            with_analysis=False
        )
        
        # Create temporary ground truth file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as gt_file:
            gt_file.write(ground_truth_data)
            gt_path = Path(gt_file.name)
        
        try:
            # Compare predictions
            comparison_results = predict_pipeline.compare_predictions(
                Path(prediction_results["predictions_file"]),
                gt_path
            )
            
            # Check evaluation results
            assert "total_samples" in comparison_results
            assert "accuracy" in comparison_results
            assert "precision" in comparison_results
            assert "recall" in comparison_results
            assert "f1_score" in comparison_results
            assert comparison_results["total_samples"] > 0
            assert 0 <= comparison_results["accuracy"] <= 1
            
        finally:
            gt_path.unlink()  # Clean up
    
    def test_pipeline_error_handling(self, temp_config: AppConfig) -> None:
        """Test error handling in pipelines."""
        from titanic_ml.pipeline.predict import PredictionPipeline
        
        # Test prediction without trained model
        predict_pipeline = PredictionPipeline(temp_config)
        
        # This should fail because model doesn't exist
        temp_config.prediction.model_path = Path("nonexistent_model.joblib")
        
        validation_results = predict_pipeline.validate_configuration()
        assert not validation_results["is_valid"]
        assert len(validation_results["errors"]) > 0