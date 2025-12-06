#!/usr/bin/env python3
"""
PostgreSQL 데이터베이스를 SQLite로 변환하는 스크립트
"""
import psycopg
import sqlite3
import os
from tqdm import tqdm
import getpass

def get_postgres_tables(pg_cursor):
    """PostgreSQL에서 모든 테이블 목록 가져오기"""
    pg_cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    return [row[0] for row in pg_cursor.fetchall()]

def get_table_schema(pg_cursor, table_name):
    """테이블 스키마 정보 가져오기"""
    pg_cursor.execute("""
        SELECT 
            column_name, 
            data_type, 
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = %s 
        ORDER BY ordinal_position;
    """, (table_name,))
    return pg_cursor.fetchall()

def postgres_to_sqlite_type(pg_type):
    """PostgreSQL 타입을 SQLite 타입으로 변환"""
    type_mapping = {
        'integer': 'INTEGER',
        'bigint': 'INTEGER',
        'smallint': 'INTEGER',
        'serial': 'INTEGER',
        'bigserial': 'INTEGER',
        'real': 'REAL',
        'double precision': 'REAL',
        'numeric': 'REAL',
        'decimal': 'REAL',
        'money': 'REAL',
        'character varying': 'TEXT',
        'varchar': 'TEXT',
        'character': 'TEXT',
        'char': 'TEXT',
        'text': 'TEXT',
        'boolean': 'INTEGER',
        'date': 'TEXT',
        'timestamp without time zone': 'TEXT',
        'timestamp with time zone': 'TEXT',
        'time': 'TEXT',
        'bytea': 'BLOB',
    }
    return type_mapping.get(pg_type.lower(), 'TEXT')

def create_sqlite_table(sqlite_cursor, table_name, schema):
    """SQLite 테이블 생성"""
    columns = []
    for col_name, data_type, is_nullable, default in schema:
        sqlite_type = postgres_to_sqlite_type(data_type)
        col_def = f'"{col_name}" {sqlite_type}'
        if is_nullable == 'NO':
            col_def += ' NOT NULL'
        columns.append(col_def)
    
    create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n  ' + ',\n  '.join(columns) + '\n);'
    sqlite_cursor.execute(create_sql)
    return create_sql

def copy_table_data(pg_cursor, sqlite_cursor, table_name):
    """테이블 데이터 복사"""
    # PostgreSQL에서 데이터 가져오기
    pg_cursor.execute(f'SELECT * FROM "{table_name}"')
    rows = pg_cursor.fetchall()
    
    if not rows:
        return 0
    
    # 컬럼 수 확인
    col_count = len(rows[0])
    placeholders = ','.join(['?' for _ in range(col_count)])
    
    # SQLite에 데이터 삽입
    insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
    
    # 배치로 삽입 (성능 향상)
    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        sqlite_cursor.executemany(insert_sql, batch)
    
    return len(rows)

def main():
    # PostgreSQL 연결 정보
    password = os.getenv('POSTGRES_PASSWORD')
    if not password:
        password = getpass.getpass("PostgreSQL 비밀번호를 입력하세요: ")
    
    pg_config = {
        'host': '192.168.200.101',
        'port': 15432,
        'dbname': 'BIRD',
        'user': 'minwoo',
        'password': password
    }
    
    # SQLite 파일 경로
    sqlite_db_path = '/data/workspace/sogang/sqlagent/data/BIRD.sqlite'
    
    print("PostgreSQL에 연결 중...")
    pg_conn = psycopg.connect(**pg_config)
    pg_cursor = pg_conn.cursor()
    
    print("SQLite 데이터베이스 생성 중...")
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_cursor = sqlite_conn.cursor()
    
    # 외래 키 제약 조건 비활성화 (데이터 삽입 속도 향상)
    sqlite_cursor.execute('PRAGMA foreign_keys = OFF;')
    sqlite_cursor.execute('PRAGMA synchronous = OFF;')
    sqlite_cursor.execute('PRAGMA journal_mode = MEMORY;')
    
    try:
        # 테이블 목록 가져오기
        print("\n테이블 목록 가져오는 중...")
        tables = get_postgres_tables(pg_cursor)
        print(f"총 {len(tables)}개의 테이블 발견")
        
        # 각 테이블 변환
        for table_name in tqdm(tables, desc="테이블 변환 중"):
            print(f"\n처리 중: {table_name}")
            
            # 스키마 가져오기
            schema = get_table_schema(pg_cursor, table_name)
            
            # SQLite 테이블 생성
            create_sql = create_sqlite_table(sqlite_cursor, table_name, schema)
            print(f"  ✓ 테이블 생성됨")
            
            # 데이터 복사
            row_count = copy_table_data(pg_cursor, sqlite_cursor, table_name)
            print(f"  ✓ {row_count:,}개 행 복사됨")
            
            # 커밋 (각 테이블마다)
            sqlite_conn.commit()
        
        print("\n✅ 모든 테이블 변환 완료!")
        print(f"SQLite 데이터베이스 위치: {sqlite_db_path}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sqlite_conn.rollback()
        raise
    
    finally:
        pg_cursor.close()
        pg_conn.close()
        sqlite_cursor.close()
        sqlite_conn.close()

if __name__ == '__main__':
    main()

