import pandas as pd
import sqlite3
import os

def get_table_columns(csv_files: set, db_name: str = 'temp.db') -> str:
    conn = sqlite3.connect(db_name)
    try:
        table_columns = ''
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                continue
                
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            pd.read_csv(csv_file).to_sql(table_name, conn, if_exists='replace', index=False)
            
            columns = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)['name'].tolist()
            table_columns += f'Table {table_name}:\nHas Columns: {",".join(columns)}\n\n'
            
        return table_columns
    finally:
        conn.close()