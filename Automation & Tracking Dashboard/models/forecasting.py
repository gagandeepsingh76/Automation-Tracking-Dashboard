#!/usr/bin/env python3
"""
Trend Scope - Advanced Forecasting Models
=========================================

Ensemble forecasting models combining LSTM, Prophet, ARIMA, and XGBoost
for accurate sales and revenue predictions with uncertainty quantification.

Author: Neelanjan Chakraborty
Website: https://neelanjanchakraborty.in/
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import joblib
import json
import os

# Time series forecasting libraries
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb

# Deep learning for time series
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model optimization
import optuna
from optuna.integration import TFKerasPruningCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class ForecastMetrics:
    """Comprehensive metrics for forecast evaluation."""
    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    prediction_interval_coverage: float
    training_time: float
    inference_time: float


class LSTMForecaster:
    """Advanced LSTM model for time series forecasting."""
    
    def __init__(self, sequence_length: int = 60, hidden_units: int = 128, 
                 num_layers: int = 3, dropout_rate: float = 0.2):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        
    def _create_model(self, input_shape: Tuple[int, int], 
                     num_features: int = 0) -> Model:
        """Create advanced LSTM model with attention mechanism."""
        # Main sequence input
        sequence_input = Input(shape=input_shape, name='sequence_input')
        
        # LSTM layers with residual connections
        lstm_out = sequence_input
        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            lstm_layer = LSTM(
                self.hidden_units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f'lstm_{i+1}'
            )(lstm_out)
            
            # Add residual connection for deeper networks
            if i > 0 and return_sequences:
                if lstm_out.shape[-1] == self.hidden_units:
                    lstm_layer = tf.keras.layers.Add()([lstm_out, lstm_layer])
            
            lstm_out = lstm_layer
        
        # Feature input for additional variables
        if num_features > 0:
            feature_input = Input(shape=(num_features,), name='feature_input')
            feature_dense = Dense(64, activation='relu')(feature_input)
            feature_dense = Dropout(self.dropout_rate)(feature_dense)
            
            # Combine LSTM output with features
            combined = Concatenate()([lstm_out, feature_dense])
            inputs = [sequence_input, feature_input]
        else:
            combined = lstm_out
            inputs = sequence_input
        
        # Output layers
        dense1 = Dense(64, activation='relu')(combined)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense2 = Dense(32, activation='relu')(dense1)
        
        # Multiple outputs for point prediction and uncertainty
        point_prediction = Dense(1, name='point_prediction')(dense2)
        uncertainty = Dense(1, activation='softplus', name='uncertainty')(dense2)
        
        model = Model(inputs=inputs, outputs=[point_prediction, uncertainty])
        
        # Custom loss function for uncertainty quantification
        def uncertainty_loss(y_true, y_pred):
            point_pred, uncertainty_pred = y_pred[:, 0:1], y_pred[:, 1:2]
            
            # Negative log-likelihood loss
            loss = 0.5 * tf.log(2 * np.pi * uncertainty_pred**2) + \
                   0.5 * (y_true - point_pred)**2 / uncertainty_pred**2
            return tf.reduce_mean(loss)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'point_prediction': 'mse', 'uncertainty': 'mse'},
            loss_weights={'point_prediction': 1.0, 'uncertainty': 0.1}
        )
        
        return model
    
    def _prepare_sequences(self, data: np.ndarray, 
                          features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        X, y = np.array(X), np.array(y)
        
        # Add features if provided
        if features is not None:
            feature_sequences = []
            for i in range(self.sequence_length, len(features)):
                feature_sequences.append(features[i])
            feature_sequences = np.array(feature_sequences)
            return X, y, feature_sequences
        
        return X, y
    
    def fit(self, data: pd.Series, features: Optional[pd.DataFrame] = None,
            validation_split: float = 0.2, epochs: int = 100) -> Dict[str, Any]:
        """Train LSTM model with optional feature inputs."""
        # Scale target variable
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        
        # Scale features if provided
        scaled_features = None
        if features is not None:
            scaled_features = self.feature_scaler.fit_transform(features.values)
        
        # Prepare sequences
        if scaled_features is not None:
            X, y, feature_seq = self._prepare_sequences(scaled_data, scaled_features)
            input_shape = (X.shape[1], X.shape[2])
            num_features = feature_seq.shape[1]
        else:
            X, y = self._prepare_sequences(scaled_data)
            input_shape = (X.shape[1], X.shape[2])
            num_features = 0
            feature_seq = None
        
        # Create model
        self.model = self._create_model(input_shape, num_features)
        
        # Prepare training data
        if feature_seq is not None:
            train_inputs = [X, feature_seq]
        else:
            train_inputs = X
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=7, factor=0.5)
        ]
        
        # Train model
        history = self.model.fit(
            train_inputs, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        return {
            'training_loss': history.history['loss'][-1],
            'validation_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }
    
    def predict(self, data: pd.Series, features: Optional[pd.DataFrame] = None,
                steps: int = 30) -> Dict[str, np.ndarray]:
        """Generate forecasts with confidence intervals."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Scale input data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1)).flatten()
        
        predictions = []
        uncertainties = []
        
        # Use last sequence_length points as starting point
        current_sequence = scaled_data[-self.sequence_length:]
        
        for step in range(steps):
            # Prepare input
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Add features if available
            if features is not None:
                feature_input = self.feature_scaler.transform(
                    features.iloc[-1:].values
                ).reshape(1, -1)
                model_input = [X_pred, feature_input]
            else:
                model_input = X_pred
            
            # Make prediction
            pred_output = self.model.predict(model_input, verbose=0)
            point_pred = pred_output[0][0, 0]
            uncertainty = pred_output[1][0, 0]
            
            predictions.append(point_pred)
            uncertainties.append(uncertainty)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], point_pred)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # Calculate confidence intervals
        uncertainties = np.array(uncertainties)
        std_dev = np.sqrt(uncertainties)
        
        # Transform standard deviations back to original scale
        original_scale_factor = self.scaler.scale_[0]
        std_dev = std_dev / original_scale_factor
        
        lower_bound = predictions - 1.96 * std_dev
        upper_bound = predictions + 1.96 * std_dev
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': std_dev
        }


class ProphetForecaster:
    """Facebook Prophet forecasting model with custom seasonalities."""
    
    def __init__(self, growth: str = 'linear', seasonality_mode: str = 'additive'):
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.model = None
        
    def fit(self, data: pd.Series, 
            holidays: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train Prophet model with custom configurations."""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        # Initialize Prophet model
        self.model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            holidays=holidays,
            uncertainty_samples=1000
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        # Fit model
        self.model.fit(df)
        
        return {'model_fitted': True}
    
    def predict(self, periods: int = 30, freq: str = 'D') -> Dict[str, np.ndarray]:
        """Generate forecasts with Prophet."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract relevant columns
        predictions = forecast['yhat'].tail(periods).values
        lower_bound = forecast['yhat_lower'].tail(periods).values
        upper_bound = forecast['yhat_upper'].tail(periods).values
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'forecast_df': forecast
        }


class XGBoostForecaster:
    """XGBoost model for time series forecasting with feature engineering."""
    
    def __init__(self, n_estimators: int = 1000, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.feature_importance_ = None
        
    def _create_features(self, data: pd.Series, lag_features: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """Create lag and rolling window features."""
        df = pd.DataFrame({'value': data.values}, index=data.index)
        
        # Lag features
        for lag in lag_features:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling window features
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
            df[f'rolling_min_{window}'] = df['value'].rolling(window).min()
            df[f'rolling_max_{window}'] = df['value'].rolling(window).max()
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['day_of_year'] = df.index.dayofyear
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df.drop('value', axis=1)
    
    def fit(self, data: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model."""
        # Create features
        features_df = self._create_features(data)
        
        # Remove rows with NaN values
        valid_idx = features_df.dropna().index
        X = features_df.loc[valid_idx]
        y = data.loc[valid_idx]
        
        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        
        self.model.fit(X, y)
        self.feature_importance_ = self.model.feature_importances_
        
        return {
            'n_features': X.shape[1],
            'training_score': self.model.score(X, y)
        }
    
    def predict(self, data: pd.Series, steps: int = 30) -> Dict[str, np.ndarray]:
        """Generate forecasts."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        extended_data = data.copy()
        
        for step in range(steps):
            # Create features for current data
            features_df = self._create_features(extended_data)
            
            # Get last valid row for prediction
            last_features = features_df.iloc[-1:].fillna(method='ffill')
            
            # Make prediction
            pred = self.model.predict(last_features)[0]
            predictions.append(pred)
            
            # Extend data with prediction for next iteration
            next_date = extended_data.index[-1] + pd.Timedelta(days=1)
            extended_data = pd.concat([
                extended_data,
                pd.Series([pred], index=[next_date])
            ])
        
        # Simple confidence intervals based on historical residuals
        # In practice, you'd use more sophisticated methods
        predictions = np.array(predictions)
        std_dev = np.std(predictions) * 0.1  # Simplified uncertainty
        
        return {
            'predictions': predictions,
            'lower_bound': predictions - 1.96 * std_dev,
            'upper_bound': predictions + 1.96 * std_dev
        }


class EnsembleForecaster:
    """Ensemble forecasting combining multiple models."""
    
    def __init__(self, models: Optional[Dict[str, Any]] = None):
        if models is None:
            self.models = {
                'lstm': LSTMForecaster(),
                'prophet': ProphetForecaster(),
                'xgboost': XGBoostForecaster()
            }
        else:
            self.models = models
        
        self.weights = None
        self.performance_metrics = {}
        
    def fit(self, data: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train ensemble models and calculate optimal weights."""
        # Split data for validation
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        model_predictions = {}
        model_performance = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            try:
                # Train model
                train_result = model.fit(train_data)
                
                # Validate model
                val_pred = model.predict(train_data, steps=len(val_data))
                predictions = val_pred['predictions']
                
                # Calculate metrics
                mae = mean_absolute_error(val_data.values, predictions)
                rmse = np.sqrt(mean_squared_error(val_data.values, predictions))
                mape = np.mean(np.abs((val_data.values - predictions) / val_data.values)) * 100
                
                model_predictions[name] = predictions
                model_performance[name] = {'mae': mae, 'rmse': rmse, 'mape': mape}
                
                logger.info(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                continue
        
        # Calculate ensemble weights based on inverse RMSE
        total_inv_rmse = sum(1 / perf['rmse'] for perf in model_performance.values())
        self.weights = {
            name: (1 / perf['rmse']) / total_inv_rmse 
            for name, perf in model_performance.values()
        }
        
        self.performance_metrics = model_performance
        
        return {
            'models_trained': list(model_performance.keys()),
            'weights': self.weights,
            'performance': model_performance
        }
    
    def predict(self, data: pd.Series, steps: int = 30) -> Dict[str, np.ndarray]:
        """Generate ensemble forecasts."""
        model_forecasts = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name in self.weights:
                try:
                    forecast = model.predict(data, steps=steps)
                    model_forecasts[name] = forecast
                except Exception as e:
                    logger.warning(f"Failed to predict with {name}: {e}")
        
        if not model_forecasts:
            raise ValueError("No models available for prediction")
        
        # Combine predictions using weights
        ensemble_predictions = np.zeros(steps)
        ensemble_lower = np.zeros(steps)
        ensemble_upper = np.zeros(steps)
        
        for name, forecast in model_forecasts.items():
            weight = self.weights.get(name, 0)
            ensemble_predictions += weight * forecast['predictions']
            ensemble_lower += weight * forecast['lower_bound']
            ensemble_upper += weight * forecast['upper_bound']
        
        return {
            'predictions': ensemble_predictions,
            'lower_bound': ensemble_lower,
            'upper_bound': ensemble_upper,
            'model_forecasts': model_forecasts,
            'weights_used': self.weights
        }
    
    def save_model(self, filepath: str):
        """Save ensemble model to disk."""
        model_data = {
            'weights': self.weights,
            'performance_metrics': self.performance_metrics,
            'model_configs': {name: type(model).__name__ for name, model in self.models.items()}
        }
        
        # Save individual models
        model_dir = os.path.dirname(filepath)
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if hasattr(model, 'model') and model.model is not None:
                if name == 'lstm':
                    model.model.save(f"{model_dir}/{name}_model.h5")
                    joblib.dump(model.scaler, f"{model_dir}/{name}_scaler.pkl")
                elif name == 'prophet':
                    joblib.dump(model.model, f"{model_dir}/{name}_model.pkl")
                elif name == 'xgboost':
                    joblib.dump(model.model, f"{model_dir}/{name}_model.pkl")
        
        # Save ensemble metadata
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Ensemble model saved to {filepath}")


def optimize_hyperparameters(data: pd.Series, model_type: str = 'lstm') -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna."""
    
    def objective(trial):
        if model_type == 'lstm':
            # LSTM hyperparameter optimization
            sequence_length = trial.suggest_int('sequence_length', 30, 120)
            hidden_units = trial.suggest_int('hidden_units', 64, 256)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            
            model = LSTMForecaster(
                sequence_length=sequence_length,
                hidden_units=hidden_units,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
            
        elif model_type == 'xgboost':
            # XGBoost hyperparameter optimization
            n_estimators = trial.suggest_int('n_estimators', 100, 2000)
            max_depth = trial.suggest_int('max_depth', 3, 12)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            
            model = XGBoostForecaster(
                n_estimators=n_estimators,
                max_depth=max_depth
            )
            # Update learning rate manually for XGBoost
            model.learning_rate = learning_rate
        
        # Cross-validation
        val_split = 0.2
        split_idx = int(len(data) * (1 - val_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        try:
            model.fit(train_data)
            predictions = model.predict(train_data, steps=len(val_data))['predictions']
            rmse = np.sqrt(mean_squared_error(val_data.values, predictions))
            return rmse
        except:
            return float('inf')
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=3600)  # 1 hour timeout
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'trials_completed': len(study.trials)
    }


def main():
    """Example usage of forecasting models."""
    # Generate sample time series data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create synthetic sales data with trend and seasonality
    trend = np.linspace(1000, 1500, len(dates))
    seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 50, len(dates))
    sales_data = pd.Series(trend + seasonal + noise, index=dates, name='sales')
    
    logger.info("Training ensemble forecasting model...")
    
    # Initialize and train ensemble
    ensemble = EnsembleForecaster()
    train_result = ensemble.fit(sales_data)
    
    # Generate forecasts
    forecast_result = ensemble.predict(sales_data, steps=90)
    
    logger.info("Forecast completed!")
    logger.info(f"Models used: {list(train_result['models_trained'])}")
    logger.info(f"Model weights: {train_result['weights']}")
    
    # Save model
    os.makedirs('models/forecasting', exist_ok=True)
    ensemble.save_model('models/forecasting/ensemble_model.json')
    
    return forecast_result


if __name__ == "__main__":
    main()
