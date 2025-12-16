#!/usr/bin/env python
"""Tool to inspect parquet files in detail."""
import sys
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import json


def inspect_parquet(parquet_file: Path):
    """Comprehensive parquet file inspection."""
    print('='*70)
    print(f'PARQUET FILE INSPECTION: {parquet_file.name}')
    print('='*70)
    
    # ============== METHOD 1: PyArrow (Low-level schema) ==============
    print('\n📋 SCHEMA (PyArrow):')
    try:
        parquet_file_obj = pq.ParquetFile(parquet_file)
        schema = parquet_file_obj.schema_arrow  # Use arrow schema
        
        print(f'   Number of columns: {len(schema)}')
        print(f'   Number of row groups: {parquet_file_obj.num_row_groups}')
        print(f'   Metadata: {parquet_file_obj.metadata.num_rows} rows')
        
        print(f'\n   Columns:')
        for i, field in enumerate(schema):
            print(f'      {i:2d}. {field.name:30s} | {str(field.type):20s}')
        
        # Check metadata
        print(f'\n   Schema metadata:')
        if schema.metadata:
            for key, value in schema.metadata.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value
                print(f'      {key_str}: {value_str[:100]}...' if len(value_str) > 100 else f'      {key_str}: {value_str}')
        else:
            print('      (none)')
    except Exception as e:
        print(f'   Error reading schema: {e}')
        print(f'   Continuing with pandas only...')
    
    # ============== METHOD 2: Pandas (Data inspection) ==============
    print(f'\n📊 DATA (Pandas):')
    df = pd.read_parquet(parquet_file)
    
    print(f'   Shape: {df.shape} (rows, columns)')
    print(f'   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB')
    
    print(f'\n   Columns with dtypes:')
    for col, dtype in df.dtypes.items():
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f'      {col:30s} | {str(dtype):20s} | {non_null:6d} non-null, {null_count:6d} null')
    
    # ============== METHOD 3: Sample Data ==============
    print(f'\n📝 SAMPLE DATA (first 3 rows):')
    print(df.head(3).to_string())
    
    # ============== METHOD 4: Episode Analysis ==============
    if 'episode_index' in df.columns:
        print(f'\n📈 EPISODE BREAKDOWN:')
        episodes = df['episode_index'].unique()
        print(f'   Episodes: {sorted(episodes)}')
        
        for ep in sorted(episodes):
            ep_data = df[df['episode_index'] == ep]
            print(f'\n   Episode {ep}:')
            print(f'      Rows: {len(ep_data)}')
            print(f'      Columns: {list(ep_data.columns)}')
            
            if 'timestamp' in ep_data.columns:
                print(f'      Duration: {ep_data["timestamp"].max():.2f}s')
            
            if 'index' in ep_data.columns:
                print(f'      Index range: {ep_data["index"].min()} to {ep_data["index"].max()}')
            
            # Check for video columns
            video_cols = [col for col in ep_data.columns if 'image' in col.lower() or 'video' in col.lower()]
            print(f'      Video columns: {video_cols if video_cols else "NONE"}')
    
    # ============== METHOD 5: Check for Video References ==============
    print(f'\n🎥 VIDEO COLUMN CHECK:')
    video_related = [col for col in df.columns if any(x in col.lower() for x in ['image', 'video', 'camera', 'rgb'])]
    
    if video_related:
        print(f'   Found video-related columns: {video_related}')
        for col in video_related:
            print(f'\n   Column: {col}')
            print(f'      Sample values:')
            for i in range(min(3, len(df))):
                print(f'         [{i}]: {df[col].iloc[i]}')
    else:
        print(f'   ❌ NO VIDEO-RELATED COLUMNS FOUND!')
        print(f'   This explains why our viewer cant load videos.')
    
    # ============== METHOD 6: Raw Column Dump ==============
    print(f'\n💾 SAVE DETAILED DUMP:')
    output_json = parquet_file.parent / f'{parquet_file.stem}_dump.json'
    
    dump_data = {
        'file': str(parquet_file),
        'schema': {
            'columns': [{'name': field.name, 'type': str(field.type)} for field in schema],
            'num_rows': parquet_file_obj.metadata.num_rows,
            'num_row_groups': parquet_file_obj.num_row_groups,
        },
        'data': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample': df.head(5).to_dict(orient='records'),
        }
    }
    
    if 'episode_index' in df.columns:
        dump_data['episodes'] = {
            int(ep): {
                'rows': int((df['episode_index'] == ep).sum()),
                'duration': float(df[df['episode_index'] == ep]['timestamp'].max()) if 'timestamp' in df.columns else None,
            }
            for ep in sorted(df['episode_index'].unique())
        }
    
    with open(output_json, 'w') as f:
        json.dump(dump_data, f, indent=2, default=str)
    
    print(f'   Saved detailed dump to: {output_json}')
    
    return df


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print('Usage: python inspect_parquet.py <parquet_file>')
        print('\nExample:')
        print('  python inspect_parquet.py ~/.cache/huggingface/lerobot/dataset/pick_and_place_small_cube/data/chunk-000/file-000.parquet')
        sys.exit(1)
    
    parquet_file = Path(sys.argv[1])
    
    if not parquet_file.exists():
        print(f'❌ File not found: {parquet_file}')
        sys.exit(1)
    
    df = inspect_parquet(parquet_file)
    
    print(f'\n{"="*70}')
    print('INSPECTION COMPLETE')
    print(f'{"="*70}')
    
    # Interactive mode
    print(f'\n💡 TIP: DataFrame is available as "df" if you want to explore more:')
    print(f'   df.info()')
    print(f'   df.describe()')
    print(f'   df["column_name"].value_counts()')
    
    # Check if running interactively
    import __main__
    if hasattr(__main__, '__file__'):
        # Running as script
        pass
    else:
        # Running in IPython/Jupyter
        return df


if __name__ == '__main__':
    main()

