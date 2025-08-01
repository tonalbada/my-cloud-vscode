import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from io import StringIO

def get_latest_atp_rankings():
    """Hämtar senaste ATP-rankings från Jeff Sackmanns GitHub"""

    github_api_url = "https://api.github.com/repos/JeffSackmann/tennis_atp/contents"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(github_api_url, headers=headers)
        response.raise_for_status()
        files = response.json()

        ranking_files = [f for f in files if f['name'].startswith('atp_rankings_') and f['name'].endswith('.csv')]
        if not ranking_files:
            print("❌ Inga ranking-filer hittades")
            return None, None

        latest_file = sorted(ranking_files, key=lambda x: x['name'])[-1]
        print(f"📊 Hämtar senaste rankings: {latest_file['name']}")

        file_response = requests.get(latest_file['download_url'], headers=headers)
        file_response.raise_for_status()

        rankings_df = pd.read_csv(StringIO(file_response.text))
        return rankings_df, latest_file['name']

    except Exception as e:
        print(f"❌ Fel vid hämtning av rankings: {e}")
        return None, None


def get_player_names():
    """Hämtar spelarnamn från Jeff Sackmanns GitHub"""

    players_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(players_url, headers=headers)
        response.raise_for_status()

        players_df = pd.read_csv(StringIO(response.text))
        print(f"👥 Hämtade {len(players_df)} spelarnamn")
        return players_df

    except Exception as e:
        print(f"⚠️  Kunde inte hämta spelarnamn: {e}")
        return None


def scrape_tennis_abstract_elo():
    """Hämtar ELO-data från Tennis Abstract med HTML-tabeller"""

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    url = "https://tennisabstract.com/reports/atp_elo_ratings.html"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        tables = soup.find_all('table')
        print(f"📊 Hittade {len(tables)} HTML-tabeller")

        best_table = None
        max_rows = 0

        for i, table in enumerate(tables):
            rows = table.find_all('tr')
            print(f"  Tabell {i+1}: {len(rows)} rader")

            if rows:
                first_row = rows[0]
                cells = first_row.find_all(['td', 'th'])
                cell_texts = [cell.get_text().strip() for cell in cells]
                print(f"    Kolumner: {cell_texts[:5]}")

                if any('elo' in str(cell).lower() for cell in cell_texts):
                    if len(rows) > max_rows:
                        max_rows = len(rows)
                        best_table = table
                        print(f"    ✅ Bästa ELO-tabell hittills: {len(rows)} rader")

        if not best_table:
            print("❌ Ingen ELO-tabell hittades")
            return None

        print(f"📈 Använder tabell med {max_rows} rader")
        rows = best_table.find_all('tr')
        data = []

        for row in rows[1:]:
            cells = row.find_all(['td', 'th'])
            row_data = []
            for cell in cells:
                text = cell.get_text().strip()
                text = text.replace('\xa0', ' ').replace('\n', ' ').strip()
                row_data.append(text)

            if len(row_data) >= 4:
                data.append(row_data)

        if not data:
            print("❌ Ingen data extraherad från tabellen")
            return None

        print(f"📈 Extraherade {len(data)} datarader")

        columns = ['elo_rank', 'player_name', 'age', 'elo']
        max_cols = max(len(row) for row in data) if data else 4
        while len(columns) < max_cols:
            columns.append(f'col_{len(columns)}')
        min_cols = min(len(columns), min(len(row) for row in data))

        df = pd.DataFrame([row[:min_cols] for row in data], columns=columns[:min_cols])
        print(f"✅ DataFrame skapad med kolumner: {list(df.columns)}")
        print(f"📊 Första raden: {df.iloc[0].to_dict()}")

        numeric_columns = ['elo_rank', 'age', 'elo']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'player_name' in df.columns:
            df['player_name'] = df['player_name'].str.strip()

        return df

    except Exception as e:
        print(f"❌ Fel vid scraping av Tennis Abstract: {e}")
        import traceback
        traceback.print_exc()
        return None


def combine_elo_with_rankings():
    """Huvudfunktion som kombinerar ELO-data med ATP-rankings"""

    print("🎾 Hämtar komplett ATP ELO + ranking data...\n")

    print("1️⃣ Hämtar ELO-data från Tennis Abstract...")
    elo_df = scrape_tennis_abstract_elo()
    if elo_df is None:
        return None

    print("\n2️⃣ Hämtar senaste ATP-rankings...")
    rankings_result = get_latest_atp_rankings()
    if rankings_result[0] is None:
        print("⚠️  Fortsätter utan ATP-rankings matchning")
        rankings_df, rankings_file = None, None
    else:
        rankings_df, rankings_file = rankings_result

    print("\n3️⃣ Hämtar spelarnamn...")
    players_df = get_player_names()

    elo_df['last_updated'] = datetime.today().strftime('%Y-%m-%d')

    if 'elo_rank' in elo_df.columns:
        elo_df = elo_df.sort_values('elo_rank', na_position='last')

    print(f"\n✅ Färdig! {len(elo_df)} spelare med ELO-data")
    return elo_df


if __name__ == "__main__":
    result_df = combine_elo_with_rankings()

    if result_df is not None:
        print(f"\n🏆 TOP 10 ELO RANKINGS:")
        print("=" * 80)

        for i in range(min(10, len(result_df))):
            row = result_df.iloc[i]
            rank = int(row.get('elo_rank', i+1)) if pd.notna(row.get('elo_rank')) else i+1
            name = row.get('player_name', 'Okänt namn')
            elo = row.get('elo', 0)
            age = row.get('age', 0)

            print(f"{rank:2d}. {name:<25} ELO: {elo:>6} | Age: {age:>2}")

        filename = "atp_elo_rankings_latest.csv"
        result_df.to_csv(filename, index=False)
        print(f"\n💾 Sparad till: {filename}")
        print(f"📊 Totalt {len(result_df)} spelare med ELO-data")
        print(f"📋 Kolumner: {list(result_df.columns)}")

    else:
        print("❌ Kunde inte hämta data. Försök igen.")