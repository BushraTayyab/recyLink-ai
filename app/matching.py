from app.models import Supply, Demand, Recommendation
from datetime import datetime


def material_matches(supply: Supply, demand: Demand) -> bool:
    return supply.material == demand.material

import math

def calculate_distance(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    radius_of_earth_km = 6371

    latitude1 = math.radians(latitude1)
    latitude2 = math.radians(latitude2)

    delta_latitude = latitude2 - latitude1
    delta_longitude = math.radians(longitude2 - longitude1)

    a = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude1)
        * math.cos(latitude2)
        * math.sin(delta_longitude / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_of_earth_km * c

def within_distance(supply: Supply, demand: Demand, max_distance_km: float = 10) -> bool:
    distance = calculate_distance(
        supply.latitude,
        supply.longitude,
        demand.latitude,
        demand.longitude,
    )

    return distance <= max_distance_km

def is_match(supply: Supply, demand: Demand) -> bool:
    return (
        material_matches(supply, demand)
        and within_distance(supply, demand)
        and is_active(demand)
    )

def find_matches(supply: Supply, demands: list[Demand]) -> list[Demand]:
    matches = []

    for demand in demands:
        if is_match(supply, demand):
            matches.append(demand)

    return matches
    
def is_active(demand: Demand) -> bool:
    return datetime.now() < demand.expires_at

def calculate_price_score(price: float, min_price: float, max_price: float) -> float:
    if max_price == min_price:
        return 1.0

    return (price - min_price) / (max_price - min_price)

def calculate_distance_score(
    distance: float,
    min_distance: float,
    max_distance: float,
) -> float:
    if max_distance == min_distance:
        return 1.0

    return (max_distance - distance) / (max_distance - min_distance)

def calculate_final_score(
    price_score: float,
    distance_score: float,
) -> float:
    price_weight = 0.6
    distance_weight = 0.4

    return (
        price_weight * price_score
        + distance_weight * distance_score
    )
    
def rank_matches(
    supply: Supply,
    matches: list[Demand],
) -> list[Recommendation]:

    if not matches:
        return []

    prices = [demand.offered_price for demand in matches]

    distances = [
        calculate_distance(
            supply.latitude,
            supply.longitude,
            demand.latitude,
            demand.longitude,
        )
        for demand in matches
    ]

    min_price = min(prices)
    max_price = max(prices)

    min_distance = min(distances)
    max_distance = max(distances)

    recommendations = []

    for demand, distance in zip(matches, distances):

        price_score = calculate_price_score(
            demand.offered_price,
            min_price,
            max_price,
        )

        distance_score = calculate_distance_score(
            distance,
            min_distance,
            max_distance,
        )

        final_score = calculate_final_score(
            price_score,
            distance_score,
        )

        recommendations.append(
            Recommendation(
                demand=demand,
                distance_km=distance,
                price_score=price_score,
                distance_score=distance_score,
                final_score=final_score,
            )
        )

    recommendations.sort(
        key=lambda recommendation: recommendation.final_score,
        reverse=True,
    )

    return recommendations