"""
State-of-the-Art Random Forest Tennis Predictor
Baserat på etablerad forskning (Gao & Kowalczyk 2021, Wilkens 2021)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from joblib import dump, load, Parallel, delayed
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gc

warnings.filterwarnings('ignore')


class StateOfArtTennisRF:
    """State-of-the-art Random Forest för tennisprediction"""
    
    def __init__(self, elo_integration=None, verbose=True):
        self.rf_model = None
        self.feature_importance = None
        self.scaler = RobustScaler()
        self.imputer = None
        self.elo_integration = elo_integration
        self.feature_names = None
        self.optimal_features = None
        self.feature_stats = {}
        self.verbose = verbose
        
        self.base_params = {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': 'balanced',
            'oob_score': True
        }
    
    def _safe_divide(self, numerator, denominator, default=0):
        """Säker division som hanterar 0 och NaN"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                (denominator != 0) & (~np.isnan(denominator)) & (~np.isnan(numerator)),
                numerator / denominator,
                default
            )
        return result
    
    def prepare_features(self, df, add_elo=True):
        """Förbereder features enligt forskningens rekommendationer"""
        if self.verbose:
            print("🔧 Förbereder features enligt forskningen...")
        
        data = df.copy()
        
        if 'tourney_date' in data.columns:
            data['tourney_date'] = pd.to_datetime(data['tourney_date'], errors='coerce')
        
        # 1. SERVE-FEATURES
        if self.verbose:
            print("  📊 Skapar serve-features...")
        
        data['serve_win_pct'] = self._safe_divide(
            data['1stWon'].fillna(0) + data['2ndWon'].fillna(0),
            data['svpt'].fillna(1),
            default=0.5
        )
        
        data['serve_win_pct_rolling'] = self._safe_divide(
            data['1stWon_rolling_10'].fillna(0) + data['2ndWon_rolling_10'].fillna(0),
            data['svpt_rolling_10'].fillna(1),
            default=0.5
        )
        
        data['1st_serve_win_pct'] = self._safe_divide(
            data['1stWon'].fillna(0),
            data['1stIn'].fillna(1),
            default=0.65
        )
        
        data['1st_serve_in_pct'] = self._safe_divide(
            data['1stIn'].fillna(0),
            data['svpt'].fillna(1),
            default=0.6
        )
        
        second_serves = data['svpt'].fillna(0) - data['1stIn'].fillna(0)
        data['2nd_serve_win_pct'] = self._safe_divide(
            data['2ndWon'].fillna(0),
            second_serves,
            default=0.5
        )
        
        data['ace_rate'] = self._safe_divide(
            data['ace'].fillna(0),
            data['svpt'].fillna(1),
            default=0.05
        )
        
        data['df_rate'] = self._safe_divide(
            data['df'].fillna(0),
            data['svpt'].fillna(1),
            default=0.05
        )
        
        # 2. RETURN-FEATURES
        if self.verbose:
            print("  📊 Skapar return-features...")
        
        data['bp_save_rate'] = self._safe_divide(
            data['bpSaved'].fillna(0),
            data['bpFaced'].fillna(1),
            default=0.6
        )
        
        data['bp_conversion'] = 1 - data['bp_save_rate']
        
        if 'bpSaved_rolling_10_opponent' in data.columns and 'bpFaced_rolling_10_opponent' in data.columns:
            data['bp_conversion_opp'] = 1 - self._safe_divide(
                data['bpSaved_rolling_10_opponent'].fillna(0),
                data['bpFaced_rolling_10_opponent'].fillna(1),
                default=0.6
            )
        
        # 3. FORM-FEATURES
        if self.verbose:
            print("  📊 Skapar form-features...")
        
        for window in [5, 20]:
            col_name = f'outcome_rolling_{window}'
            if col_name not in data.columns and 'outcome' in data.columns:
                data[col_name] = (
                    data.groupby('player_name')['outcome']
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
        
        # 4. MOMENTUM/STREAK
        if self.verbose:
            print("  📊 Beräknar momentum...")
        
        data['current_streak'] = self._calculate_streak(data)
        
        if 'outcome_rolling_10' in data.columns:
            data['hot_streak'] = (data['outcome_rolling_10'] > 0.7).astype(int)
            data['cold_streak'] = (data['outcome_rolling_10'] < 0.3).astype(int)
        
        # 5. UNDERLAGSSPECIFIKA FEATURES
        if 'surface' in data.columns:
            if self.verbose:
                print("  📊 Skapar underlagsspecifika features...")
            
            for surface in data['surface'].unique():
                if pd.notna(surface):
                    col_name = f'{surface.lower()}_win_rate'
                    surface_mask = data['surface'] == surface
                    
                    data[col_name] = data.groupby('player_name').apply(
                        lambda x: x[surface_mask]['outcome'].expanding().mean()
                    ).reset_index(level=0, drop=True)
                    
                    data[col_name] = data[col_name].fillna(0.5)
            
            surface_dummies = pd.get_dummies(data['surface'], prefix='surface')
            data = pd.concat([data, surface_dummies], axis=1)
        
        # 6. RANK-BASERADE FEATURES
        if self.verbose:
            print("  📊 Skapar rank-features...")
        
        data['player_rank_clean'] = data['player_rank'].fillna(500).clip(1, 2000)
        data['opponent_rank_clean'] = data['opponent_rank'].fillna(500).clip(1, 2000)
        
        data['player_rank_log'] = np.log1p(data['player_rank_clean'])
        data['opponent_rank_log'] = np.log1p(data['opponent_rank_clean'])
        data['rank_diff_log'] = data['player_rank_log'] - data['opponent_rank_log']
        
        data['player_rank_category'] = pd.cut(
            data['player_rank_clean'],
            bins=[0, 10, 20, 50, 100, 200, float('inf')],
            labels=['top10', 'top20', 'top50', 'top100', 'top200', 'outside200']
        )
        
        # 7. INTERAKTIONSFEATURES
        if self.verbose:
            print("  📊 Skapar interaktionsfeatures...")
        
        if 'surface' in data.columns:
            surface_numeric = pd.Categorical(data['surface']).codes
            data['serve_surface_interaction'] = data['serve_win_pct_rolling'] * (surface_numeric + 1)
        
        if 'rank_diff' in data.columns and 'outcome_rolling_10' in data.columns:
            data['rank_form_interaction'] = data['rank_diff'] * data['outcome_rolling_10']
        
        # 8. ELO-FEATURES
        if add_elo and self.elo_integration is not None:
            if self.verbose:
                print("  📊 Lägger till ELO-features...")
            data = self._add_elo_features(data)
        
        # 9. DIFFERENTIELLA FEATURES
        if self.verbose:
            print("  📊 Skapar differentiella features...")
        
        if 'serve_win_pct_rolling' in data.columns and 'serve_win_pct_rolling_opponent' not in data.columns:
            avg_serve_win = data['serve_win_pct_rolling'].mean()
            data['serve_win_pct_rolling_opponent'] = avg_serve_win
        
        diff_features = [
            ('serve_win_pct_rolling', 'serve_win_pct_rolling_opponent'),
            ('outcome_rolling_10', 'outcome_rolling_10_opponent'),
            ('ace_rolling_10', 'ace_rolling_10_opponent'),
            ('df_rolling_10', 'df_rolling_10_opponent'),
            ('1stWon_rolling_10', '1stWon_rolling_10_opponent'),
            ('2ndWon_rolling_10', '2ndWon_rolling_10_opponent')
        ]
        
        for feat, opp_feat in diff_features:
            if feat in data.columns and opp_feat in data.columns:
                diff_name = f"{feat.replace('_rolling_10', '').replace('_rolling', '')}_advantage"
                data[diff_name] = data[feat] - data[opp_feat]
        
        # 10. FATIGUE
        if self.verbose:
            print("  📊 Beräknar trötthetsfaktorer...")
        
        data['matches_last_7_days'] = self._calculate_recent_matches(data, days=7)
        data['matches_last_30_days'] = self._calculate_recent_matches(data, days=30)
        
        # 11. CLEANUP
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        if self.verbose:
            print("  ✅ Feature preparation klar!")
        
        return data
    
    def _calculate_streak(self, df):
        """Beräkna current winning/losing streak"""
        if 'outcome' not in df.columns:
            return pd.Series(0, index=df.index)
        
        streaks = []
        
        for player_name in df['player_name'].unique():
            player_mask = df['player_name'] == player_name
            player_data = df[player_mask].sort_values('tourney_date')
            
            player_streaks = []
            current_streak = 0
            
            for outcome in player_data['outcome']:
                if pd.isna(outcome):
                    player_streaks.append(0)
                    continue
                
                if outcome == 1:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                else:
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                
                player_streaks.append(current_streak)
            
            player_indices = df[player_mask].sort_values('tourney_date').index
            for idx, streak in zip(player_indices, player_streaks):
                streaks.append((idx, streak))
        
        streaks.sort(key=lambda x: x[0])
        return pd.Series([s[1] for s in streaks], index=[s[0] for s in streaks])
    
    def _calculate_recent_matches(self, df, days=7):
        """Beräkna antal matcher inom X dagar"""
        if 'tourney_date' not in df.columns:
            return pd.Series(0, index=df.index)
        
        matches_count = pd.Series(index=df.index, dtype=int)
        
        for idx, row in df.iterrows():
            if pd.isna(row['tourney_date']):
                matches_count[idx] = 0
                continue
            
            player_matches = df[
                (df['player_name'] == row['player_name']) & 
                (df['tourney_date'] >= row['tourney_date'] - timedelta(days=days)) &
                (df['tourney_date'] < row['tourney_date']) &
                (df.index != idx)
            ]
            
            matches_count[idx] = len(player_matches)
        
        return matches_count
    
    def _add_elo_features(self, df):
        """Integrera ELO-data"""
        if self.elo_integration is None:
            return df
        
        elo_columns = ['player_elo', 'opponent_elo', 'elo_diff', 'elo_win_prob']
        for col in elo_columns:
            df[col] = np.nan
        
        unique_players = df['player_name'].unique()
        player_elo_cache = {}
        
        for player in unique_players:
            elo_data = self.elo_integration.get_elo_features(player)
            if elo_data:
                player_elo_cache[player] = elo_data
        
        for idx, row in df.iterrows():
            player_name = row['player_name']
            opponent_name = row['opponent_name']
            
            if player_name in player_elo_cache:
                df.loc[idx, 'player_elo'] = player_elo_cache[player_name].get('elo_rating', np.nan)
            
            if opponent_name in player_elo_cache:
                df.loc[idx, 'opponent_elo'] = player_elo_cache[opponent_name].get('elo_rating', np.nan)
        
        df['elo_diff'] = df['player_elo'] - df['opponent_elo']
        df['elo_win_prob'] = 1 / (1 + 10**(-df['elo_diff']/400))
        
        return df
    
    def select_features(self, df, target_col='outcome', top_k=35):
        """Feature selection baserat på importance"""
        if self.verbose:
            print("\n🎯 Väljer optimala features...")
        
        exclude_cols = [
            'outcome', 'player_name', 'opponent_name', 'tourney_date',
            'player_id', 'opponent_id', 'surface'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        nan_threshold = 0.5
        valid_features = []
        for feat in numeric_features:
            nan_ratio = df[feat].isna().sum() / len(df)
            if nan_ratio < nan_threshold:
                valid_features.append(feat)
        
        if self.verbose:
            print(f"  📊 {len(valid_features)} features efter NaN-filtrering")
        
        if len(valid_features) < 20:
            nan_threshold = 0.7
            valid_features = []
            for feat in numeric_features:
                nan_ratio = df[feat].isna().sum() / len(df)
                if nan_ratio < nan_threshold:
                    valid_features.append(feat)
        
        X_temp = df[valid_features].fillna(df[valid_features].median())
        y_temp = df[target_col]
        
        mask = ~y_temp.isna()
        X_temp = X_temp[mask]
        y_temp = y_temp[mask]
        
        rf_quick = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_quick.fit(X_temp, y_temp)
        
        importance_df = pd.DataFrame({
            'feature': valid_features,
            'importance': rf_quick.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if self.verbose:
            print(f"\n🏆 Top 20 features enligt importance:")
            print(importance_df.head(20).to_string(index=False))
        
        selected_features = []
        
        serve_keywords = ['serve', '1st', '2nd', 'ace', 'svpt', 'SvGms']
        serve_features = []
        for _, row in importance_df.iterrows():
            if any(keyword in row['feature'] for keyword in serve_keywords):
                serve_features.append(row['feature'])
                if len(serve_features) >= 8:
                    break
        selected_features.extend(serve_features)
        
        rank_keywords = ['rank', 'elo']
        rank_features = []
        for _, row in importance_df.iterrows():
            if any(keyword in row['feature'] for keyword in rank_keywords):
                if row['feature'] not in selected_features:
                    rank_features.append(row['feature'])
                    if len(rank_features) >= 5:
                        break
        selected_features.extend(rank_features)
        
        form_keywords = ['rolling', 'streak', 'outcome']
        form_features = []
        for _, row in importance_df.iterrows():
            if any(keyword in row['feature'] for keyword in form_keywords):
                if row['feature'] not in selected_features:
                    form_features.append(row['feature'])
                    if len(form_features) >= 5:
                        break
        selected_features.extend(form_features)
        
        diff_keywords = ['diff', 'advantage']
        diff_features = []
        for _, row in importance_df.iterrows():
            if any(keyword in row['feature'] for keyword in diff_keywords):
                if row['feature'] not in selected_features:
                    diff_features.append(row['feature'])
                    if len(diff_features) >= 5:
                        break
        selected_features.extend(diff_features)
        
        surface_features = [f for f in valid_features if f.startswith('surface_')]
        selected_features.extend([f for f in surface_features if f not in selected_features][:3])
        
        for _, row in importance_df.iterrows():
            if row['feature'] not in selected_features:
                selected_features.append(row['feature'])
                if len(selected_features) >= top_k:
                    break
        
        selected_features = selected_features[:top_k]
        
        if self.verbose:
            print(f"\n✅ Valde {len(selected_features)} features för modellen")
        
        self.optimal_features = selected_features
        self.feature_importance = importance_df
        
        return selected_features
    
    def train(self, df, optimize_hyperparams=True, cv_folds=5, test_size=0.2):
        """Träna modellen"""
        if self.verbose:
            print("\n🚀 Tränar State-of-the-Art Random Forest...")
        
        df_processed = self.prepare_features(df)
        selected_features = self.select_features(df_processed)
        self.feature_names = selected_features
        
        X = df_processed[selected_features]
        y = df_processed['outcome']
        
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if 'tourney_date' in df_processed.columns:
            dates = df_processed.loc[mask, 'tourney_date']
            sorted_idx = dates.argsort()
            X = X.iloc[sorted_idx]
            y = y.iloc[sorted_idx]
        
        if self.verbose:
            print(f"\n📊 Träningsdata: {X.shape[0]} matcher, {X.shape[1]} features")
        
        nan_ratio = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
        if nan_ratio > 0.3:
            self.imputer = SimpleImputer(strategy='median')
            if self.verbose:
                print(f"   Använder median imputation (NaN ratio: {nan_ratio:.1%})")
        else:
            self.imputer = KNNImputer(n_neighbors=5)
            if self.verbose:
                print(f"   Använder KNN imputation (NaN ratio: {nan_ratio:.1%})")
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if self.verbose:
            print(f"   Temporal split: {len(X_train)} train, {len(X_test)} test")
        
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        for i, feat in enumerate(self.feature_names):
            self.feature_stats[feat] = {
                'mean': X_train_imputed[:, i].mean(),
                'std': X_train_imputed[:, i].std(),
                'median': np.median(X_train_imputed[:, i])
            }
        
        if optimize_hyperparams:
            if self.verbose:
                print("\n🔍 Optimerar hyperparametrar...")
            self.rf_model = self._optimize_hyperparameters(X_train_scaled, y_train, cv_folds)
        else:
            if self.verbose:
                print("\n🏗️ Tränar med forskningsbaserade hyperparametrar...")
            self.rf_model = RandomForestClassifier(**self.base_params)
            self.rf_model.fit(X_train_scaled, y_train)
            if hasattr(self.rf_model, 'oob_score_'):
                print(f"   OOB Score: {self.rf_model.oob_score_:.3f}")
        
        if self.verbose:
            print("\n📊 Utvärderar på test set...")
        
        y_pred = self.rf_model.predict(X_test_scaled)
        y_pred_proba = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        if self.verbose:
            print(f"   Test Accuracy: {test_metrics['accuracy']:.3f}")
            print(f"   Test Log Loss: {test_metrics['log_loss']:.3f}")
            print(f"   Test Brier Score: {test_metrics['brier_score']:.3f}")
            print(f"   Test AUC: {test_metrics['auc']:.3f}")
        
        cv_scores = None
        if cv_folds > 1:
            if self.verbose:
                print(f"\n📊 Cross-validation på träningsdata ({cv_folds} folds)...")
            cv_scores = self._evaluate_model(X_train_scaled, y_train, cv_folds)
        
        return self.rf_model, {
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'n_features': len(self.feature_names),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def _optimize_hyperparameters(self, X, y, cv_folds=5):
        """Grid search för hyperparameter optimization"""
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [15, 20, 25, 30],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.3]
        }
        
        tscv = TimeSeriesSplit(n_splits=min(cv_folds, 5))
        
        grid_search = GridSearchCV(
            RandomForestClassifier(
                bootstrap=True,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced',
                oob_score=True
            ),
            param_grid,
            cv=tscv,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1 if self.verbose else 0
        )
        
        grid_search.fit(X, y)
        
        if self.verbose:
            print(f"\n🏆 Bästa hyperparametrar:")
            for param, value in grid_search.best_params_.items():
                print(f"   {param}: {value}")
            print(f"   Best CV Log Loss: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, X, y, cv_folds=5):
        """Utvärdera modellen med flera metrics"""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        metrics = {
            'accuracy': [],
            'log_loss': [],
            'brier_score': [],
            'auc': []
        }
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            temp_rf = RandomForestClassifier(**self.base_params)
            temp_rf.fit(X_train, y_train)
            
            y_pred = temp_rf.predict(X_test)
            y_pred_proba = temp_rf.predict_proba(X_test)[:, 1]
            
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['log_loss'].append(log_loss(y_test, y_pred_proba))
            metrics['brier_score'].append(brier_score_loss(y_test, y_pred_proba))
            metrics['auc'].append(roc_auc_score(y_test, y_pred_proba))
        
        if self.verbose:
            for metric, values in metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"   {metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")
        
        return metrics
    
    def predict(self, player1, player2, surface=None, df=None):
        """Gör prediction för en match"""
        if self.rf_model is None:
            raise ValueError("Modellen måste tränas först!")
        
        if df is None:
            raise ValueError("DataFrame krävs för prediction")
        
        p1_data = df[df['player_name'] == player1].sort_values('tourney_date', ascending=False).head(1)
        p2_data = df[df['player_name'] == player2].sort_values('tourney_date', ascending=False).head(1)
        
        if p1_data.empty:
            return {'error': f"Ingen data för {player1}"}
        if p2_data.empty:
            return {'error': f"Ingen data för {player2}"}
        
        features = self._create_match_features(
            p1_data.iloc[0], 
            p2_data.iloc[0], 
            surface
        )
        
        feature_array = np.array([features[feat] for feat in self.feature_names])
        
        feature_array = feature_array.reshape(1, -1)
        features_imputed = self.imputer.transform(feature_array)
        features_scaled = self.scaler.transform(features_imputed)
        
        prob = self.rf_model.predict_proba(features_scaled)[0, 1]
        
        tree_predictions = np.array([tree.predict_proba(features_scaled)[0, 1] 
                                   for tree in self.rf_model.estimators_])
        prediction_std = np.std(tree_predictions)
        
        return {
            'player1': player1,
            'player2': player2,
            'player1_win_prob': prob,
            'player2_win_prob': 1 - prob,
            'confidence': 1 - prediction_std,
            'surface': surface
        }
    
    def _create_match_features(self, p1_data, p2_data, surface):
        """Skapa feature dictionary för en match"""
        features = {}
        
        for feat in self.feature_names:
            if feat in p1_data.index:
                features[feat] = p1_data[feat]
            elif feat.endswith('_opponent'):
                base_feat = feat.replace('_opponent', '')
                if base_feat in p2_data.index:
                    features[feat] = p2_data[base_feat]
                else:
                    features[feat] = self.feature_stats.get(feat, {}).get('mean', 0)
            elif feat.endswith('_diff') or feat.endswith('_advantage'):
                base_feat = feat.replace('_diff', '').replace('_advantage', '')
                if base_feat in p1_data.index and base_feat in p2_data.index:
                    features[feat] = p1_data[base_feat] - p2_data[base_feat]
                else:
                    features[feat] = 0
            elif feat.startswith('surface_'):
                if surface and feat == f'surface_{surface}':
                    features[feat] = 1
                else:
                    features[feat] = 0
            elif '_interaction' in feat:
                if 'serve_surface_interaction' in feat:
                    serve_val = p1_data.get('serve_win_pct_rolling', 0.5)
                    surface_val = 1 if surface else 0
                    features[feat] = serve_val * surface_val
                elif 'rank_form_interaction' in feat:
                    rank_diff = p1_data.get('rank_diff', 0)
                    form = p1_data.get('outcome_rolling_10', 0.5)
                    features[feat] = rank_diff * form
                else:
                    features[feat] = 0
            else:
                features[feat] = self.feature_stats.get(feat, {}).get('mean', 0)
        
        return features
    
    def plot_feature_importance(self, top_n=25):
        """Visualisera feature importance"""
        if self.feature_importance is None:
            print("❌ Kör train() först för att beräkna feature importance")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - State of Art RF')
        plt.gca().invert_yaxis()
        
        colors = []
        for feat in top_features['feature']:
            if any(x in feat for x in ['serve', '1st', '2nd', 'ace', 'svpt']):
                colors.append('darkgreen')
            elif any(x in feat for x in ['rank', 'elo']):
                colors.append('darkblue')
            elif 'rolling' in feat or 'streak' in feat:
                colors.append('darkred')
            else:
                colors.append('gray')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='Serve'),
            Patch(facecolor='darkblue', label='Rank/ELO'),
            Patch(facecolor='darkred', label='Form'),
            Patch(facecolor='gray', label='Other')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('feature_importance_state_of_art.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Sparade feature importance plot")
    
    def save_model(self, filepath='tennis_rf_state_of_art.joblib'):
        """Spara modellen och alla komponenter"""
        model_package = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'optimal_features': self.optimal_features,
            'feature_stats': self.feature_stats,
            'version': 'StateOfArt-2.0',
            'created': datetime.now().isoformat()
        }
        
        dump(model_package, filepath)
        print(f"💾 Sparade modell till {filepath}")
    
    def load_model(self, filepath='tennis_rf_state_of_art.joblib'):
        """Ladda modell från fil"""
        model_package = load(filepath)
        
        self.rf_model = model_package['rf_model']
        self.scaler = model_package['scaler']
        self.imputer = model_package['imputer']
        self.feature_names = model_package['feature_names']
        self.feature_importance = model_package['feature_importance']
        self.optimal_features = model_package['optimal_features']
        self.feature_stats = model_package.get('feature_stats', {})
        
        print(f"✅ Laddade modell från {filepath}")
        print(f"   Version: {model_package.get('version', 'Unknown')}")
        print(f"   Features: {len(self.feature_names)}")


if __name__ == "__main__":
    print("🎾 State-of-the-Art Tennis Random Forest")
    print("=" * 60)
    
    print("\n📂 Laddar data...")
    df = load("symmetric_df_latest.joblib")
    print(f"✅ Laddade {len(df)} rader")
    
    try:
        from elo_integration import ELOIntegration
        elo = ELOIntegration()
        print("✅ ELO-integration laddad")
    except:
        elo = None
        print("⚠️  Kör utan ELO-integration")
    
    rf_model = StateOfArtTennisRF(elo_integration=elo, verbose=True)
    
    print("\n🚀 Startar träning...")
    model, scores = rf_model.train(
        df, 
        optimize_hyperparams=False,
        cv_folds=3,
        test_size=0.2
    )
    
    rf_model.plot_feature_importance(top_n=25)
    rf_model.save_model('tennis_rf_state_of_art_complete.joblib')
    
    print("\n✅ Modell tränad och sparad!")
    print(f"📊 Features: {len(rf_model.optimal_features)}")
    print(f"🎯 Test metrics: {scores['test_metrics']}")
    
    print("\n🎾 Testar prediktioner:")
    print("-" * 50)
    
    test_matches = [
        ("Novak Djokovic", "Carlos Alcaraz", "Hard"),
        ("Rafael Nadal", "Casper Ruud", "Clay"),
        ("Jannik Sinner", "Daniil Medvedev", "Hard")
    ]
    
    for p1, p2, surface in test_matches:
        pred = rf_model.predict(p1, p2, surface, df)
        if 'error' not in pred:
            print(f"\n{p1} vs {p2} ({surface}):")
            print(f"  {p1}: {pred['player1_win_prob']*100:.1f}%")
            print(f"  {p2}: {pred['player2_win_prob']*100:.1f}%")
            print(f"  Confidence: {pred['confidence']:.2f}")
        else:
            print(f"\n{p1} vs {p2}: {pred['error']}")