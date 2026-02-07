import pandas as pd
import logging
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Integer, Float, Date, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, date
from typing import List, Dict, Optional
import os
import hashlib

logging.basicConfig(level=logging.INFO)


class TenantManager:
    """
    Multi-tenant management system with data isolation
    Supports multiple plants/factories with separate data streams
    """
    
    def __init__(self, engine=None):
        """
        Initialize tenant manager
        
        Args:
            engine: SQLAlchemy engine (if None, creates new one)
        """
        if engine is None:
            user = os.getenv('DB_USER', 'mfg_user')
            password = os.getenv('DB_PASSWORD', 'mfg_pass123')
            host = os.getenv('DB_HOST', 'db')
            port = os.getenv('DB_PORT', '5432')
            dbname = os.getenv('DB_NAME', 'manufacturing')
            
            self.engine = create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}")
        else:
            self.engine = engine
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.metadata = MetaData()
        
        # Initialize tenant tables
        self.init_tenant_schema()
    
    def init_tenant_schema(self):
        """
        Create multi-tenant database schema
        
        Schema Design:
        - tenants: Master table of all plants/tenants
        - production_data_tenant: Production data with tenant_id for isolation
        - machines_tenant: Machine catalog per tenant
        - users_tenant: User access control per tenant
        """
        try:
            with self.engine.connect() as conn:
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
                logging.info("Multi-tenant schema initialized successfully")
                
        except Exception as e:
            logging.error(f"Tenant schema initialization failed: {e}")
            raise
    
    def create_tenant(self, tenant_id: str, tenant_name: str, plant_location: str = None, 
                     timezone: str = 'UTC', config: dict = None) -> bool:
        """
        Create a new tenant (plant/factory)
        
        Args:
            tenant_id: Unique identifier for tenant
            tenant_name: Display name
            plant_location: Physical location
            timezone: Timezone for the plant
            config: Additional configuration (JSON)
            
        Returns:
            Success boolean
        """
        try:
            with self.engine.connect() as conn:
                config_json = config if config else {}
                
                conn.execute(text("""
                    INSERT INTO tenants (tenant_id, tenant_name, plant_location, timezone, config)
                    VALUES (:tenant_id, :tenant_name, :plant_location, :timezone, :config::jsonb)
                    ON CONFLICT (tenant_id) DO UPDATE 
                    SET tenant_name = :tenant_name, 
                        plant_location = :plant_location,
                        timezone = :timezone,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    'tenant_id': tenant_id,
                    'tenant_name': tenant_name,
                    'plant_location': plant_location,
                    'timezone': timezone,
                    'config': str(config_json).replace("'", '"')
                })
                
                conn.commit()
                logging.info(f"Tenant created/updated: {tenant_id} - {tenant_name}")
                return True
                
        except Exception as e:
            logging.error(f"Tenant creation failed: {e}")
            return False
    
    def get_tenant_list(self, active_only: bool = True) -> List[Dict]:
        """
        Get list of all tenants
        
        Args:
            active_only: Only return active tenants
            
        Returns:
            List of tenant dictionaries
        """
        try:
            query = "SELECT * FROM tenants"
            if active_only:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY tenant_name"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                tenants = [dict(row._mapping) for row in result]
                
            logging.info(f"Retrieved {len(tenants)} tenants")
            return tenants
            
        except Exception as e:
            logging.error(f"Failed to retrieve tenant list: {e}")
            return []
    
    def add_production_data(self, tenant_id: str, production_data: List[Dict]) -> bool:
        """
        Add production data for a specific tenant with isolation
        
        Args:
            tenant_id: Tenant identifier
            production_data: List of production records
            
        Returns:
            Success boolean
        """
        try:
            if not production_data:
                return True
            
            # Add tenant_id to each record
            for record in production_data:
                record['tenant_id'] = tenant_id
            
            df = pd.DataFrame(production_data)
            
            # Ensure required columns
            required_cols = ['tenant_id', 'production_date', 'machine_id', 
                           'units_produced', 'defective_units', 'downtime_min', 'shift']
            
            if not all(col in df.columns for col in required_cols):
                logging.error(f"Missing required columns. Need: {required_cols}")
                return False
            
            # Insert to tenant-isolated table
            with self.engine.connect() as conn:
                df.to_sql('production_data_tenant', conn, if_exists='append', index=False)
                conn.commit()
                
            logging.info(f"Added {len(production_data)} records for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add production data for tenant {tenant_id}: {e}")
            return False
    
    def get_production_data(self, tenant_id: str, days: int = 7, 
                           machine_id: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve production data for a specific tenant (isolated)
        
        Args:
            tenant_id: Tenant identifier
            days: Number of days of history
            machine_id: Optional filter by machine
            
        Returns:
            DataFrame with production data
        """
        try:
            query = f"""
                SELECT production_date, machine_id, units_produced, 
                       defective_units, downtime_min, shift, created_at
                FROM production_data_tenant
                WHERE tenant_id = :tenant_id
                AND production_date >= CURRENT_DATE - INTERVAL '{days} days'
            """
            
            params = {'tenant_id': tenant_id}
            
            if machine_id:
                query += " AND machine_id = :machine_id"
                params['machine_id'] = machine_id
            
            query += " ORDER BY production_date DESC, machine_id"
            
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params)
            
            logging.info(f"Retrieved {len(df)} production records for tenant {tenant_id}")
            return df
            
        except Exception as e:
            logging.error(f"Failed to retrieve production data for tenant {tenant_id}: {e}")
            return pd.DataFrame()
    
    def register_machine(self, tenant_id: str, machine_id: str, machine_name: str,
                        machine_type: str = None, capacity: float = None) -> bool:
        """
        Register a machine for a tenant
        
        Args:
            tenant_id: Tenant identifier
            machine_id: Machine identifier (unique within tenant)
            machine_name: Display name
            machine_type: Type of machine
            capacity: Production capacity (units/hour)
            
        Returns:
            Success boolean
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO machines_tenant 
                    (tenant_id, machine_id, machine_name, machine_type, capacity_units_per_hour)
                    VALUES (:tenant_id, :machine_id, :machine_name, :machine_type, :capacity)
                    ON CONFLICT (tenant_id, machine_id) DO UPDATE
                    SET machine_name = :machine_name,
                        machine_type = :machine_type,
                        capacity_units_per_hour = :capacity
                """), {
                    'tenant_id': tenant_id,
                    'machine_id': machine_id,
                    'machine_name': machine_name,
                    'machine_type': machine_type,
                    'capacity': capacity
                })
                
                conn.commit()
                logging.info(f"Machine registered: {machine_id} for tenant {tenant_id}")
                return True
                
        except Exception as e:
            logging.error(f"Machine registration failed: {e}")
            return False
    
    def get_tenant_machines(self, tenant_id: str, active_only: bool = True) -> List[Dict]:
        """
        Get all machines for a tenant
        
        Args:
            tenant_id: Tenant identifier
            active_only: Only return active machines
            
        Returns:
            List of machine dictionaries
        """
        try:
            query = "SELECT * FROM machines_tenant WHERE tenant_id = :tenant_id"
            if active_only:
                query += " AND is_active = TRUE"
            query += " ORDER BY machine_id"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'tenant_id': tenant_id})
                machines = [dict(row._mapping) for row in result]
            
            logging.info(f"Retrieved {len(machines)} machines for tenant {tenant_id}")
            return machines
            
        except Exception as e:
            logging.error(f"Failed to retrieve machines for tenant {tenant_id}: {e}")
            return []
    
    def grant_user_access(self, user_id: str, tenant_id: str, user_name: str,
                         user_email: str, role: str = 'viewer') -> bool:
        """
        Grant user access to a tenant
        
        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            user_name: User display name
            user_email: User email
            role: User role (admin, manager, viewer)
            
        Returns:
            Success boolean
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO users_tenant (user_id, tenant_id, user_name, user_email, role)
                    VALUES (:user_id, :tenant_id, :user_name, :user_email, :role)
                    ON CONFLICT (user_id, tenant_id) DO UPDATE
                    SET user_name = :user_name,
                        user_email = :user_email,
                        role = :role
                """), {
                    'user_id': user_id,
                    'tenant_id': tenant_id,
                    'user_name': user_name,
                    'user_email': user_email,
                    'role': role
                })
                
                conn.commit()
                logging.info(f"User access granted: {user_id} to tenant {tenant_id} as {role}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to grant user access: {e}")
            return False
    
    def get_user_tenants(self, user_id: str) -> List[Dict]:
        """
        Get all tenants a user has access to
        
        Args:
            user_id: User identifier
            
        Returns:
            List of tenant dictionaries with user role
        """
        try:
            query = """
                SELECT t.*, ut.role, ut.user_name, ut.user_email
                FROM tenants t
                JOIN users_tenant ut ON t.tenant_id = ut.tenant_id
                WHERE ut.user_id = :user_id AND ut.is_active = TRUE AND t.is_active = TRUE
                ORDER BY t.tenant_name
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'user_id': user_id})
                tenants = [dict(row._mapping) for row in result]
            
            logging.info(f"User {user_id} has access to {len(tenants)} tenants")
            return tenants
            
        except Exception as e:
            logging.error(f"Failed to retrieve user tenants: {e}")
            return []
    
    def get_cross_tenant_summary(self, days: int = 7) -> pd.DataFrame:
        """
        Get aggregated summary across all tenants (for corporate overview)
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with tenant-level summaries
        """
        try:
            query = f"""
                SELECT 
                    t.tenant_id,
                    t.tenant_name,
                    t.plant_location,
                    COUNT(DISTINCT pd.machine_id) as total_machines,
                    SUM(pd.units_produced) as total_units,
                    SUM(pd.defective_units) as total_defects,
                    AVG(pd.downtime_min) as avg_downtime,
                    ROUND(
                        (SUM(pd.units_produced) - SUM(pd.defective_units)) * 100.0 / 
                        NULLIF(SUM(pd.units_produced), 0), 2
                    ) as yield_percentage
                FROM tenants t
                LEFT JOIN production_data_tenant pd ON t.tenant_id = pd.tenant_id
                    AND pd.production_date >= CURRENT_DATE - INTERVAL '{days} days'
                WHERE t.is_active = TRUE
                GROUP BY t.tenant_id, t.tenant_name, t.plant_location
                ORDER BY total_units DESC NULLS LAST
            """
            
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn)
            
            logging.info(f"Cross-tenant summary generated for {len(df)} tenants")
            return df
            
        except Exception as e:
            logging.error(f"Failed to generate cross-tenant summary: {e}")
            return pd.DataFrame()
    
    def migrate_legacy_data(self, tenant_id: str) -> bool:
        """
        Migrate data from legacy live_production table to tenant-isolated table
        
        Args:
            tenant_id: Tenant to assign legacy data to
            
        Returns:
            Success boolean
        """
        try:
            with self.engine.connect() as conn:
                # Check if legacy table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'live_production'
                    )
                """))
                
                if not result.scalar():
                    logging.info("No legacy data to migrate")
                    return True
                
                # Migrate data
                conn.execute(text(f"""
                    INSERT INTO production_data_tenant 
                    (tenant_id, production_date, machine_id, units_produced, 
                     defective_units, downtime_min, shift)
                    SELECT 
                        '{tenant_id}' as tenant_id,
                        production_date,
                        machine_id,
                        units_produced,
                        defective_units,
                        downtime_min,
                        shift
                    FROM live_production
                    WHERE NOT EXISTS (
                        SELECT 1 FROM production_data_tenant pt
                        WHERE pt.tenant_id = '{tenant_id}'
                        AND pt.production_date = live_production.production_date
                        AND pt.machine_id = live_production.machine_id
                    )
                """))
                
                conn.commit()
                logging.info(f"Legacy data migrated to tenant {tenant_id}")
                return True
                
        except Exception as e:
            logging.error(f"Legacy data migration failed: {e}")
            return False


# Singleton instance
_tenant_manager = None

def get_tenant_manager(engine=None) -> TenantManager:
    """Get or create singleton tenant manager instance"""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager(engine)
    return _tenant_manager