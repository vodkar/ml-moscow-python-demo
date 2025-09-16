#!/usr/bin/env python3
"""Simple demo of the Titanic ML pipeline using only sklearn models."""

import sys
import pandas as pd
from pathlib import Path

# Create a simple demo without external model dependencies
def create_demo_model():
    """Create and train a simple Random Forest model."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    
    print("🚀 Starting Titanic ML Demo")
    
    # Load data
    data_path = Path("data")
    if not (data_path / "train.csv").exists():
        print(f"❌ Training data not found at {data_path / 'train.csv'}")
        return False
    
    # Simple data loading and preprocessing
    df = pd.read_csv(data_path / "train.csv")
    print(f"📊 Loaded {len(df)} training records")
    
    # Basic feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Simple preprocessing
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    # Prepare features and target
    X = df[features]
    y = df['Survived']
    
    print(f"🔧 Features: {X.shape[1]} columns")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Training Complete!")
    print(f"📈 Test Accuracy: {accuracy:.4f}")
    print(f"\n📊 Detailed Results:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 5 Most Important Features:")
    for _, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Test with test data if available
    test_path = data_path / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        print(f"\n🔮 Making predictions on {len(test_df)} test records...")
        
        # Apply same preprocessing
        test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
        test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)
        
        test_df['Age'].fillna(df['Age'].median(), inplace=True)
        test_df['Fare'].fillna(df['Fare'].median(), inplace=True)
        test_df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        test_df['Sex'] = le_sex.transform(test_df['Sex'])
        test_df['Embarked'] = le_embarked.transform(test_df['Embarked'])
        
        X_test_new = test_df[features]
        predictions = model.predict(X_test_new)
        probabilities = model.predict_proba(X_test_new)[:, 1]
        
        # Create submission
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions
        })
        
        output_path = Path("demo_submission.csv")
        submission.to_csv(output_path, index=False)
        
        print(f"📝 Predictions saved to: {output_path}")
        print(f"🎲 Survival rate in test predictions: {predictions.mean():.3f}")
        print(f"🎯 Average prediction confidence: {probabilities.mean():.3f}")
    
    return True

if __name__ == "__main__":
    try:
        success = create_demo_model()
        if success:
            print("\n🎉 Demo completed successfully!")
            print("\n💡 This demonstrates the core ML pipeline functionality")
            print("   using only standard scikit-learn components.")
            print("\n🚀 For the full experience with advanced features,")
            print("   install additional dependencies and use the CLI:")
            print("   $ titanic-ml train --data-path data --model-type random_forest")
        else:
            print("\n❌ Demo failed - please check data availability")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)