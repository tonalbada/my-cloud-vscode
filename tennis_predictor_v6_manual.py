
# integration_example.py

# Exempel på hur du integrerar ELO med din befintliga Tennis Predictor

import sys
sys.path.append('.')  # För att hitta dina moduler

from tennis_predictor_v5 import TennisPredictor
from ELOintegration import ELOIntegration, EnhancedTennisPredictor
import pandas as pd
import numpy as np
from joblib import dump, load

class FullyIntegratedPredictor(TennisPredictor):
    """Komplett integration av ELO med befintlig predictor"""

    def __init__(self, model_path="rf_retrained_no_elo.joblib", 
                 data_path="symmetric_df_latest.joblib",
                 elo_file="atp_elo_rankings_latest.csv",
                 retrain_with_elo=False):
        super().__init__(model_path, data_path)
        self.elo_integration = ELOIntegration(elo_file)
        if retrain_with_elo:
            print("🔄 Tränar om modell med ELO-features...")
            self._retrain_model_with_elo()
        self.extended_features = self.features + [
            "elo_rating", "elo_rank", "peak_elo", "elo_from_peak"
        ]
        self.elo_cache = {}

    def _add_elo_to_dataframe(self):
        print("➕ Lägger till ELO-data till dataframe...")
        self.df['elo_rating'] = np.nan
        self.df['elo_rank'] = np.nan
        self.df['peak_elo'] = np.nan
        self.df['elo_from_peak'] = np.nan
        self.df['opponent_elo_rating'] = np.nan
        self.df['elo_diff'] = np.nan

        unique_players = self.df['player_name'].unique()
        elo_mapping = {}
        for player in unique_players:
            elo_data = self.elo_integration.match_player_to_elo(player)
            if elo_data is not None:
                elo_mapping[player] = {
                    'elo_rating': elo_data.get('elo', np.nan),
                    'elo_rank': elo_data.get('elo_rank', np.nan),
                    'peak_elo': elo_data.get('Peak Elo', np.nan)
                }

        for player, elo_info in elo_mapping.items():
            mask = self.df['player_name'] == player
            for col, value in elo_info.items():
                self.df.loc[mask, col] = value

        self.df['elo_from_peak'] = self.df['elo_rating'] - self.df['peak_elo']
        print(f"✅ ELO-data tillagd för {len(elo_mapping)} av {len(unique_players)} spelare")
        return self.df

    def _retrain_model_with_elo(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report

        self._add_elo_to_dataframe()

        print("🏋️ Förbereder träningsdata...")
        print("⚠️  Modellomträning kräver match-par data - implementera efter din datastruktur")

    def get_player_snapshot_with_elo(self, name, surface=None, days_back=365):
        snapshot, matched_name = self.get_player_snapshot(name, surface, days_back)
        if snapshot is None:
            return None, matched_name
        elo_features = self.elo_integration.get_elo_features(matched_name, surface)
        if elo_features:
            enhanced_snapshot = snapshot.copy()
            for key, value in elo_features.items():
                enhanced_snapshot[key] = value
            return enhanced_snapshot, matched_name
        return snapshot, matched_name

    def predict_match_full(self, p1_name, p2_name, surface=None, 
                          simulations=1000, best_of=3, 
                          use_elo=True, elo_weight=0.25,
                          **kwargs):
        base_result = self.predict_match_enhanced(
            p1_name, p2_name, surface, simulations, best_of, **kwargs
        )
        if "error" in base_result or not use_elo:
            return base_result

        elo_features = self.elo_integration.create_elo_differential_features(
            p1_name, p2_name, surface
        )
        if elo_features and 'elo_win_prob' in elo_features:
            elo_prob = elo_features['elo_win_prob']
            p1, p2 = base_result["players"]
            base_result['predictions']['elo'] = {
                p1: elo_prob,
                p2: 1 - elo_prob
            }
            current_final = base_result['predictions']['final'][p1]
            final_with_elo = (1 - elo_weight) * current_final + elo_weight * elo_prob
            base_result['predictions']['final_with_elo'] = {
                p1: final_with_elo,
                p2: 1 - final_with_elo
            }
            base_result['elo_stats'] = {
                'elo_diff': elo_features.get('elo_rating_diff', None),
                'elo_category': elo_features.get('elo_category', None),
                'p1_elo': elo_features.get('elo_rating_diff', 0) / 2,
                'p2_elo': -elo_features.get('elo_rating_diff', 0) / 2
            }
            base_result['weights'] = {
                'rf': base_result['weight_rf'] * (1 - elo_weight),
                'simulation': (1 - base_result['weight_rf']) * (1 - elo_weight),
                'elo': elo_weight
            }
        return base_result


def print_enhanced_results(result):
    """Utökad resultatutskrift med ELO-information"""
    if "error" in result:
        print(f"❌ Fel: {result['error']}")
        return

    p1, p2 = result["players"]
    preds = result["predictions"]

    print(f"\n🎾 === TENNIS PREDIKTION MED ELO ===")
    print(f"🏟️  {p1} vs {p2} ({result['surface']})")
    print(f"\n📊 PREDIKTIONER:")

    if 'final_with_elo' in preds:
        print(f"  🎯 FINAL (med ELO):")
        print(f"     {p1}: {preds['final_with_elo'][p1]*100:.1f}%")
        print(f"     {p2}: {preds['final_with_elo'][p2]*100:.1f}%")

    print(f"\n  🌲 Random Forest: {p1} {preds['rf'][p1]*100:.1f}%")
    print(f"  🎲 Simulering: {p1} {preds['simulation'][p1]*100:.1f}%")

    if 'elo' in preds:
        print(f"  📈 ELO: {p1} {preds['elo'][p1]*100:.1f}%")

    print(f"  💫 Original final: {p1} {preds['final'][p1]*100:.1f}%")

    if 'elo_stats' in result:
        elo_stats = result['elo_stats']
        print(f"\n📊 ELO-STATISTIK:")
        if elo_stats['elo_diff']:
            print(f"  ELO-differens: {elo_stats['elo_diff']:.0f}")
            print(f"  Kategori: {elo_stats['elo_category']}")

    if 'weights' in result:
        weights = result['weights']
        print(f"\n⚖️  MODELLVIKTER:")
        for model, weight in weights.items():
            print(f"  {model}: {weight:.1%}")


if __name__ == "__main__":
    print("🎾 Fully Integrated Tennis Predictor")
    print("=" * 60)

    predictor = FullyIntegratedPredictor()

    test_matches = [
        ("Carlos Alcaraz", "Jannik Sinner", "hard"),
        ("Novak Djokovic", "Taylor Fritz", "hard"),
        ("Casper Ruud", "Stefanos Tsitsipas", "clay"),
    ]

    for p1, p2, surface in test_matches:
        print(f"\n{'='*60}")
        result = predictor.predict_match_full(
            p1, p2, 
            surface=surface,
            simulations=1000,
            use_elo=True,
            elo_weight=0.3,
            use_micro_macro=True
        )
        print_enhanced_results(result)

    predictor.elo_integration.save_name_mapping()



if __name__ == "__main__":
    model = FullyIntegratedPredictor()

    # Välj spelare manuellt
    p1 = input("Ange spelare 1: ")
    p2 = input("Ange spelare 2: ")
    surface = input("Underlag (hard/clay/grass): ").lower().strip()

    result = model.predict_match_full(p1, p2, surface=surface, simulations=10000)
    print_enhanced_results(result)
