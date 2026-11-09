"""
DataSentinel AI - Initial Data Inspection
Uses our CSVLoader to inspect raw table sizes, columns, and initial nulls.
"""
from src.ingestion.loader import CSVLoader

def main():
    loader = CSVLoader(data_dir="data/raw")
    print("Loading core tables via CSVLoader...\n" + "=" * 50)
    
    tables = loader.load_all_core_tables()
    
    for name, df in tables.items():
        meta = loader.get_metadata(df, source_name=name)
        null_count = df.isnull().sum().sum()
        print(f"\n📊 Table: [{name.upper()}]")
        print(f"   • Rows: {meta['row_count']:,} | Columns: {meta['col_count']} | Memory: {meta['memory_usage_mb']} MB")
        print(f"   • Columns: {', '.join(meta['columns'][:5])}... (+ {max(0, len(meta['columns']) - 5)} more)")
        print(f"   • Total Missing Values (Nulls): {null_count:,}")

if __name__ == "__main__":
    main()