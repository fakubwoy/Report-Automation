import os
from sqlalchemy import create_engine, text
import time
import logging

logging.basicConfig(level=logging.INFO)

def init_database():
    """Initialize PostgreSQL database with required tables"""
    user = os.getenv('DB_USER', 'mfg_user')
    password = os.getenv('DB_PASSWORD', 'mfg_pass123')
    host = os.getenv('DB_HOST', 'db')
    port = os.getenv('DB_PORT', '5432')
    dbname = os.getenv('DB_NAME', 'manufacturing')
    
    # Wait for PostgreSQL to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            engine = create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}")
            with engine.connect() as conn:
                # Create legacy table if it doesn't exist
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS live_production (
                        id SERIAL PRIMARY KEY,
                        production_date DATE NOT NULL,
                        machine_id VARCHAR(50) NOT NULL,
                        units_produced NUMERIC NOT NULL,
                        defective_units NUMERIC NOT NULL,
                        downtime_min NUMERIC NOT NULL,
                        shift VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create tenants table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS tenants (
                        tenant_id VARCHAR(50) PRIMARY KEY,
                        tenant_name VARCHAR(200) NOT NULL,
                        plant_location VARCHAR(200),
                        timezone VARCHAR(50) DEFAULT 'UTC',
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        config JSONB DEFAULT '{}'::jsonb
                    )
                """))
                
                # Create tenant-isolated production data table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS production_data_tenant (
                        id SERIAL PRIMARY KEY,
                        tenant_id VARCHAR(50) NOT NULL,
                        production_date DATE NOT NULL,
                        machine_id VARCHAR(50) NOT NULL,
                        units_produced NUMERIC NOT NULL,
                        defective_units NUMERIC NOT NULL,
                        downtime_min NUMERIC NOT NULL,
                        shift VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
                    )
                """))
                
                # Create index for fast tenant queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_production_tenant 
                    ON production_data_tenant(tenant_id, production_date DESC)
                """))
                
                # Create machines catalog per tenant
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS machines_tenant (
                        id SERIAL PRIMARY KEY,
                        tenant_id VARCHAR(50) NOT NULL,
                        machine_id VARCHAR(50) NOT NULL,
                        machine_name VARCHAR(200),
                        machine_type VARCHAR(100),
                        capacity_units_per_hour NUMERIC,
                        is_active BOOLEAN DEFAULT TRUE,
                        installed_date DATE,
                        last_maintenance_date DATE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                        UNIQUE(tenant_id, machine_id)
                    )
                """))
                
                # Create user access control
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users_tenant (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) NOT NULL,
                        tenant_id VARCHAR(50) NOT NULL,
                        user_name VARCHAR(200),
                        user_email VARCHAR(200),
                        role VARCHAR(50) DEFAULT 'viewer',
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
                        UNIQUE(user_id, tenant_id)
                    )
                """))
                
                conn.commit()
                logging.info("Database initialized successfully (including multi-tenant tables)")
                return
        except Exception as e:
            if i < max_retries - 1:
                logging.warning(f"Database connection attempt {i+1}/{max_retries} failed: {e}")
                time.sleep(2)
            else:
                logging.error(f"Failed to initialize database after {max_retries} attempts")
                raise

if __name__ == "__main__":
    init_database()