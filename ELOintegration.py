
import pandas as pd
import numpy as np
from difflib import get_close_matches
from joblib import load, dump
import os
from datetime import datetime

class ELOIntegration:
    """Klass för att integrera ELO-rankings med befintlig Tennis Predictor"""

    def __init__(self, elo_file="atp_elo_rankings_latest.csv"):
        self.elo_df = self._load_elo_data(elo_file)
        self.name_mapping_cache = {}
        self._build_name_index()

    def _load_elo_data(self, elo_file):
        if not os.path.exists(elo_file):
            raise FileNotFoundError(f"ELO-fil {elo_file} hittades inte")
        df = pd.read_csv(elo_file)
        print(f"✅ Laddade {len(df)} spelare med ELO-data")
        df['player_name_clean'] = df['player_name'].str.strip().str.lower()
        elo_cols = ['elo', 'hElo', 'cElo', 'gElo', 'Peak Elo']
        for col in elo_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _build_name_index(self):
        self.name_index = {}
        for idx, row in self.elo_df.iterrows():
            clean_name = row['player_name_clean']
            self.name_index[clean_name] = idx
            parts = clean_name.split()
            if len(parts) >= 2:
                self.name_index[f"{parts[-1]} {' '.join(parts[:-1])}"] = idx
                self.name_index[parts[-1]] = idx

    def match_player_to_elo(self, player_name, use_cache=True):
        if use_cache and player_name in self.name_mapping_cache:
            return self.name_mapping_cache[player_name]
        clean_name = player_name.strip().lower()
        if clean_name in self.name_index:
            idx = self.name_index[clean_name]
            result = self.elo_df.iloc[idx]
            self.name_mapping_cache[player_name] = result
            return result
        all_names = self.elo_df['player_name_clean'].tolist()
        for cutoff in [0.85, 0.70, 0.55]:
            matches = get_close_matches(clean_name, all_names, n=1, cutoff=cutoff)
            if matches:
                matched_name = matches[0]
                idx = self.elo_df[self.elo_df['player_name_clean'] == matched_name].index[0]
                result = self.elo_df.iloc[idx]
                self.name_mapping_cache[player_name] = result
                print(f"🔍 Fuzzy match: '{player_name}' → '{result['player_name']}' (cutoff: {cutoff})")
                return result
        parts = clean_name.split()
        if parts:
            lastname = parts[-1]
            candidates = self.elo_df[self.elo_df['player_name_clean'].str.contains(lastname, na=False)]
            if len(candidates) == 1:
                result = candidates.iloc[0]
                self.name_mapping_cache[player_name] = result
                print(f"🔍 Efternamns-match: '{player_name}' → '{result['player_name']}'")
                return result
        print(f"❌ Ingen ELO-data hittad för: {player_name}")
        return None

    def get_elo_features(self, player_name, surface=None):
        elo_data = self.match_player_to_elo(player_name)
        if elo_data is None:
            return None
        features = {
            'elo_rating': elo_data.get('elo', np.nan),
            'elo_rank': elo_data.get('elo_rank', np.nan),
            'peak_elo': elo_data.get('Peak Elo', np.nan),
            'player_age': elo_data.get('age', np.nan),
            'atp_rank': elo_data.get('ATP Rank', np.nan)
        }
        if surface:
            surface_map = {'hard': 'hElo', 'clay': 'cElo', 'grass': 'gElo'}
            surface_col = surface_map.get(surface.lower())
            if surface_col and surface_col in elo_data.index:
                features[f'{surface}_elo'] = elo_data.get(surface_col, np.nan)
                features[f'{surface}_elo_rank'] = elo_data.get(f'{surface_col[0]}Elo Rank', np.nan)
        if not pd.isna(features['elo_rating']) and not pd.isna(features['peak_elo']):
            features['elo_from_peak'] = features['elo_rating'] - features['peak_elo']
        return features

    def create_elo_differential_features(self, player1_name, player2_name, surface=None):
        p1_features = self.get_elo_features(player1_name, surface)
        p2_features = self.get_elo_features(player2_name, surface)
        if p1_features is None or p2_features is None:
            return None
        diff_features = {}
        for key in ['elo_rating', 'elo_rank', 'peak_elo', 'player_age']:
            if key in p1_features and key in p2_features:
                p1_val = p1_features[key]
                p2_val = p2_features[key]
                if not pd.isna(p1_val) and not pd.isna(p2_val):
                    diff_features[f'{key}_diff'] = p1_val - p2_val
        if surface:
            surface_elo_key = f'{surface}_elo'
            if surface_elo_key in p1_features and surface_elo_key in p2_features:
                p1_surf_elo = p1_features[surface_elo_key]
                p2_surf_elo = p2_features[surface_elo_key]
                if not pd.isna(p1_surf_elo) and not pd.isna(p2_surf_elo):
                    diff_features['surface_elo_diff'] = p1_surf_elo - p2_surf_elo
        if 'elo_rating_diff' in diff_features:
            elo_diff = diff_features['elo_rating_diff']
            diff_features['elo_win_prob'] = 1 / (1 + 10**(-elo_diff / 400))
            if abs(elo_diff) < 50:
                diff_features['elo_category'] = 'close'
            elif abs(elo_diff) < 150:
                diff_features['elo_category'] = 'moderate'
            else:
                diff_features['elo_category'] = 'large'
        return diff_features

    def update_player_features(self, existing_features, player_name, surface=None):
        elo_features = self.get_elo_features(player_name, surface)
        if elo_features is None:
            return existing_features
        updated_features = existing_features.copy()
        for key, value in elo_features.items():
            if not pd.isna(value):
                updated_features[f'elo_{key}'] = value
        return updated_features

    def save_name_mapping(self, filename="elo_name_mapping.joblib"):
        dump(self.name_mapping_cache, filename)
        print(f"💾 Sparade namn-mappning till {filename}")

    def load_name_mapping(self, filename="elo_name_mapping.joblib"):
        if os.path.exists(filename):
            self.name_mapping_cache = load(filename)
            print(f"📂 Laddade namn-mappning från {filename}")


# Enhanced predictor (förutsätter TennisPredictor finns)
class EnhancedTennisPredictor:
    """Utökad version av TennisPredictor med ELO-integration"""

    def __init__(self, model_path="rf_retrained_no_elo.joblib",
                 data_path="symmetric_df_latest.joblib",
                 elo_file="atp_elo_rankings_latest.csv"):
        self.base_predictor = TennisPredictor(model_path, data_path)
        self.elo_integration = ELOIntegration(elo_file)
        self.extended_features = self.base_predictor.features + [
            "elo_rating_diff", "elo_rank_diff", "elo_win_prob"
        ]
        self.use_elo = True

    def predict_match_with_elo(self, p1_name, p2_name, surface=None, **kwargs):
        base_result = self.base_predictor.predict_match_enhanced(p1_name, p2_name, surface, **kwargs)
        if "error" in base_result:
            return base_result
        if self.use_elo:
            elo_diff_features = self.elo_integration.create_elo_differential_features(p1_name, p2_name, surface)
            if elo_diff_features:
                base_result['elo_features'] = elo_diff_features
                if 'elo_win_prob' in elo_diff_features:
                    elo_prob = elo_diff_features['elo_win_prob']
                    elo_weight = 0.3
                    current_p1_prob = base_result['predictions']['final'][p1_name]
                    new_p1_prob = (1 - elo_weight) * current_p1_prob + elo_weight * elo_prob
                    base_result['predictions']['elo'] = {
                        p1_name: elo_prob,
                        p2_name: 1 - elo_prob
                    }
                    base_result['predictions']['final_with_elo'] = {
                        p1_name: new_p1_prob,
                        p2_name: 1 - new_p1_prob
                    }
        return base_result


if __name__ == "__main__":
    elo = ELOIntegration()
    test_names = ["Alcaraz", "Sinner", "Djokovic", "nadal", "Roger Federer"]
    print("🧪 Testar ELO-matchning:")
    print("=" * 50)
    for name in test_names:
        elo_data = elo.match_player_to_elo(name)
        if elo_data is not None:
            print(f"{name} → {elo_data['player_name']}: ELO {elo_data['elo']}")

    print("Testar differential features:")
    print("=" * 50)
    diff_features = elo.create_elo_differential_features("Alcaraz", "Sinner", "hard")
    if diff_features:
        for key, value in diff_features.items():
            print(f"{key}: {value}")
