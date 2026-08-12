"""Exports à partir du registre : CSV brut, statistiques par espèce.

Pas de carte (pas de GPS dans le projet, expérimental).

Usage:
    python export.py --db outputs/registry.db --output-dir outputs/exports
"""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def load_individuals(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM individuals", conn)
    conn.close()
    return df


def species_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["species", "n_individuals", "first_seen_at", "last_seen_at"])
    return (
        df.groupby("species")
        .agg(n_individuals=("id", "count"), first_seen_at=("first_seen_at", "min"), last_seen_at=("last_seen_at", "max"))
        .reset_index()
        .sort_values("n_individuals", ascending=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="Base SQLite du registre")
    parser.add_argument("--output-dir", required=True, help="Dossier de sortie")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_individuals(args.db)

    individuals_csv = output_dir / "individuals.csv"
    df.to_csv(individuals_csv, index=False)

    stats_csv = output_dir / "species_stats.csv"
    species_stats(df).to_csv(stats_csv, index=False)

    print(f"{len(df)} individus exportés")
    print(f"-> {individuals_csv}")
    print(f"-> {stats_csv}")


if __name__ == "__main__":
    main()
