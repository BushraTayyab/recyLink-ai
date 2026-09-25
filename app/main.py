from fastapi import FastAPI

from app.models import Supply, Demand
from app.matching import find_matches, rank_matches


app = FastAPI(title="RecyLink Matching Service")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/match")
def match_supply(
    supply: Supply,
    demands: list[Demand],
):
    matches = find_matches(supply, demands)

    recommendations = rank_matches(
        supply,
        matches,
    )

    return {
        "recommendations": [
            {
                "demand_id": recommendation.demand.id,
                "recycler_id": recommendation.demand.recycler_id,
                "material": recommendation.demand.material,
                "offered_price": recommendation.demand.offered_price,
                "distance_km": round(recommendation.distance_km, 2),
                "price_score": round(recommendation.price_score, 3),
                "distance_score": round(recommendation.distance_score, 3),
                "final_score": round(recommendation.final_score, 3),
            }
            for recommendation in recommendations
        ]
    }