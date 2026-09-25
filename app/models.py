from datetime import datetime

from pydantic import BaseModel


class Supply(BaseModel):
    material: str
    quantity_kg: float
    latitude: float
    longitude: float
    created_at: datetime


class Demand(BaseModel):
    id: str
    recycler_id: str
    material: str
    quantity_kg: float
    offered_price: float
    latitude: float
    longitude: float
    expires_at: datetime


class Recommendation(BaseModel):
    demand: Demand
    distance_km: float
    price_score: float
    distance_score: float
    final_score: float