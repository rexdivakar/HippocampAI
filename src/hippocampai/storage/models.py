"""Data models for user and session storage."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class User(BaseModel):
    """User record."""
    
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    updated_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    is_active: bool = True
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class Session(BaseModel):
    """Session record."""
    
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    last_active_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    is_active: bool = True
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None  # admin user_id who deleted
    delete_reason: Optional[str] = None
    memory_count: int = 0
    metadata: dict = Field(default_factory=dict)


class SoftDeleteRecord(BaseModel):
    """Record of soft-deleted data for admin recovery."""
    
    id: str
    entity_type: str  # 'user', 'session', 'memory'
    entity_id: str
    user_id: str
    session_id: Optional[str] = None
    deleted_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    deleted_by: str
    reason: Optional[str] = None
    original_data: dict = Field(default_factory=dict)
    is_restored: bool = False
    restored_at: Optional[datetime] = None
    restored_by: Optional[str] = None
