from datetime import datetime, timedelta

from app.models import Supply, Demand
from app.matching import find_matches, rank_matches


def create_supply():
    return Supply(
        material="copper_cable",
        quantity_kg=50,
        latitude=28.61,
        longitude=77.20,
        created_at=datetime.now(),
    )


def create_demand(
    demand_id="D001",
    material="copper_cable",
    latitude=28.61,
    longitude=77.20,
    offered_price=350,
    expires_at=None,
):
    return Demand(
        id=demand_id,
        recycler_id="R001",
        material=material,
        quantity_kg=100,
        offered_price=offered_price,
        latitude=latitude,
        longitude=longitude,
        expires_at=expires_at or datetime.now() + timedelta(days=30),
    )


def test_wrong_material_is_not_matched():
    supply = create_supply()

    demand = create_demand(
        material="laptop",
    )

    matches = find_matches(supply, [demand])

    assert matches == []


def test_demand_too_far_away_is_not_matched():
    supply = create_supply()

    demand = create_demand(
        latitude=30.00,
        longitude=77.00,
    )

    matches = find_matches(supply, [demand])

    assert matches == []


def test_expired_demand_is_not_matched():
    supply = create_supply()

    demand = create_demand(
        expires_at=datetime.now() - timedelta(days=1),
    )

    matches = find_matches(supply, [demand])

    assert matches == []


def test_valid_demand_is_matched():
    supply = create_supply()

    demand = create_demand()

    matches = find_matches(supply, [demand])

    assert len(matches) == 1
    assert matches[0].id == "D001"


def test_higher_ranked_match_comes_first():
    supply = create_supply()

    better_demand = create_demand(
        demand_id="D001",
        latitude=28.61,
        longitude=77.20,
        offered_price=350,
    )

    worse_demand = create_demand(
        demand_id="D002",
        latitude=28.62,
        longitude=77.21,
        offered_price=340,
    )

    matches = find_matches(
        supply,
        [better_demand, worse_demand],
    )

    recommendations = rank_matches(
        supply,
        matches,
    )

    assert recommendations[0].demand.id == "D001"
    assert recommendations[1].demand.id == "D002"