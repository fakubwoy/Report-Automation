# api_server_tenant_extensions.py
# Add these endpoints to your existing api_server.py

from fastapi import HTTPException, Header, Depends
from typing import Optional, List
from pydantic import BaseModel
from datetime import date
import pandas as pd

# Import tenant manager
try:
    from src.tenant_manager import get_tenant_manager
    TENANT_SUPPORT = True
except ImportError:
    TENANT_SUPPORT = False
    logging.warning("Tenant management not available")


# Pydantic models for multi-tenant API
class TenantCreate(BaseModel):
    tenant_id: str
    tenant_name: str
    plant_location: Optional[str] = None
    timezone: str = "UTC"
    config: Optional[dict] = {}


class MachineRegister(BaseModel):
    machine_id: str
    machine_name: str
    machine_type: Optional[str] = None
    capacity: Optional[float] = None


class UserAccess(BaseModel):
    user_id: str
    user_name: str
    user_email: str
    role: str = "viewer"


class ProductionDataSubmit(BaseModel):
    production_date: str
    machine_id: str
    units_produced: float
    defective_units: float
    downtime_min: float
    shift: str


# Dependency to extract tenant from header
def get_tenant_id(x_tenant_id: Optional[str] = Header(None)) -> str:
    """Extract tenant ID from request header"""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="Missing X-Tenant-ID header")
    return x_tenant_id


# Multi-Tenant API Endpoints
# Add these to your FastAPI app

@app.post("/api/tenants/create")
async def create_tenant(tenant: TenantCreate):
    """
    Create a new tenant (plant/factory)
    
    Example:
    ```
    POST /api/tenants/create
    {
        "tenant_id": "plant_chicago",
        "tenant_name": "Chicago Manufacturing Plant",
        "plant_location": "Chicago, IL, USA",
        "timezone": "America/Chicago"
    }
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        success = tm.create_tenant(
            tenant.tenant_id,
            tenant.tenant_name,
            tenant.plant_location,
            tenant.timezone,
            tenant.config
        )
        
        if success:
            return {"status": "success", "tenant_id": tenant.tenant_id}
        else:
            raise HTTPException(status_code=500, detail="Tenant creation failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tenants/list")
async def list_tenants(active_only: bool = True):
    """
    Get list of all tenants
    
    Example:
    ```
    GET /api/tenants/list?active_only=true
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        tenants = tm.get_tenant_list(active_only)
        return {"tenants": tenants, "count": len(tenants)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/machines/register")
async def register_machine(tenant_id: str, machine: MachineRegister):
    """
    Register a machine for a tenant
    
    Example:
    ```
    POST /api/tenants/plant_chicago/machines/register
    {
        "machine_id": "M001",
        "machine_name": "Assembly Line 1",
        "machine_type": "Assembly",
        "capacity": 500
    }
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        success = tm.register_machine(
            tenant_id,
            machine.machine_id,
            machine.machine_name,
            machine.machine_type,
            machine.capacity
        )
        
        if success:
            return {"status": "success", "machine_id": machine.machine_id}
        else:
            raise HTTPException(status_code=500, detail="Machine registration failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tenants/{tenant_id}/machines")
async def get_tenant_machines(tenant_id: str, active_only: bool = True):
    """
    Get all machines for a tenant
    
    Example:
    ```
    GET /api/tenants/plant_chicago/machines?active_only=true
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        machines = tm.get_tenant_machines(tenant_id, active_only)
        return {"tenant_id": tenant_id, "machines": machines, "count": len(machines)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/production/submit")
async def submit_production_data(tenant_id: str, data: List[ProductionDataSubmit]):
    """
    Submit production data for a tenant
    
    Example:
    ```
    POST /api/tenants/plant_chicago/production/submit
    [
        {
            "production_date": "2026-02-07",
            "machine_id": "M001",
            "units_produced": 450,
            "defective_units": 5,
            "downtime_min": 15,
            "shift": "Day"
        }
    ]
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        
        # Convert to list of dicts
        records = [record.dict() for record in data]
        
        success = tm.add_production_data(tenant_id, records)
        
        if success:
            return {"status": "success", "records_added": len(records)}
        else:
            raise HTTPException(status_code=500, detail="Data submission failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tenants/{tenant_id}/production")
async def get_tenant_production(tenant_id: str, days: int = 7, machine_id: Optional[str] = None):
    """
    Get production data for a tenant (isolated)
    
    Example:
    ```
    GET /api/tenants/plant_chicago/production?days=7&machine_id=M001
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        df = tm.get_production_data(tenant_id, days, machine_id)
        
        # Convert to JSON-serializable format
        data = df.to_dict('records')
        
        # Handle date serialization
        for record in data:
            if 'production_date' in record and isinstance(record['production_date'], date):
                record['production_date'] = record['production_date'].isoformat()
        
        return {
            "tenant_id": tenant_id,
            "data": data,
            "count": len(data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tenants/{tenant_id}/kpis")
async def get_tenant_kpis(tenant_id: str, days: int = 7):
    """
    Get KPIs for a specific tenant
    
    Example:
    ```
    GET /api/tenants/plant_chicago/kpis?days=7
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        df = tm.get_production_data(tenant_id, days)
        
        if df.empty:
            return {
                "tenant_id": tenant_id,
                "message": "No data available",
                "kpis": {}
            }
        
        # Rename columns to match KPI engine expectations
        df_renamed = df.rename(columns={
            'units_produced': 'Units Produced',
            'defective_units': 'Defective Units',
            'downtime_min': 'Downtime (minutes)',
            'machine_id': 'Machine ID'
        })
        
        from src.kpi_engine import calculate_kpis
        summary, machine_stats = calculate_kpis(df_renamed)
        
        return {
            "tenant_id": tenant_id,
            "kpis": summary,
            "machine_stats": machine_stats.to_dict('records') if hasattr(machine_stats, 'to_dict') else machine_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/corporate/summary")
async def get_corporate_summary(days: int = 7):
    """
    Get aggregated summary across all tenants (corporate overview)
    
    Example:
    ```
    GET /api/corporate/summary?days=7
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        df = tm.get_cross_tenant_summary(days)
        
        summary_data = df.to_dict('records')
        
        return {
            "summary": summary_data,
            "total_tenants": len(summary_data),
            "period_days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/users/grant-access")
async def grant_user_access(tenant_id: str, user: UserAccess):
    """
    Grant user access to a tenant
    
    Example:
    ```
    POST /api/tenants/plant_chicago/users/grant-access
    {
        "user_id": "john_doe",
        "user_name": "John Doe",
        "user_email": "john@company.com",
        "role": "manager"
    }
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        success = tm.grant_user_access(
            user.user_id,
            tenant_id,
            user.user_name,
            user.user_email,
            user.role
        )
        
        if success:
            return {"status": "success", "user_id": user.user_id, "tenant_id": tenant_id}
        else:
            raise HTTPException(status_code=500, detail="User access grant failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/{user_id}/tenants")
async def get_user_tenants(user_id: str):
    """
    Get all tenants a user has access to
    
    Example:
    ```
    GET /api/users/john_doe/tenants
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        tenants = tm.get_user_tenants(user_id)
        
        return {
            "user_id": user_id,
            "tenants": tenants,
            "count": len(tenants)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tenants/{tenant_id}/migrate-legacy-data")
async def migrate_legacy_data(tenant_id: str):
    """
    Migrate legacy data from live_production table to tenant-isolated table
    
    Example:
    ```
    POST /api/tenants/plant_chicago/migrate-legacy-data
    ```
    """
    if not TENANT_SUPPORT:
        raise HTTPException(status_code=501, detail="Tenant management not enabled")
    
    try:
        tm = get_tenant_manager()
        success = tm.migrate_legacy_data(tenant_id)
        
        if success:
            return {"status": "success", "message": f"Legacy data migrated to tenant {tenant_id}"}
        else:
            raise HTTPException(status_code=500, detail="Migration failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))