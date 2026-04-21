"""Pytest coverage for the TechCorp metadata filtering demo."""

from uuid import uuid4

import pytest


def _load_techcorp_data(client, collection_name):
    response = client.post("/load-techcorp-data", params={"collection_name": collection_name})
    assert response.status_code == 200
    return response.json()


def _search(client, collection_name, query, filters=None, k=5):
    payload = {
        "query": query,
        "k": k,
        "collection_name": collection_name,
    }
    if filters:
        payload["filter_metadata"] = filters

    response = client.post("/search", json=payload)
    assert response.status_code == 200
    return response.json()["results"]


@pytest.fixture
def techcorp_collection_name():
    return f"techcorp-test-{uuid4().hex[:8]}"


def test_load_techcorp_data(client, techcorp_collection_name):
    result = _load_techcorp_data(client, techcorp_collection_name)

    assert result["indexed_count"] == 10
    assert result["chunk_count"] == 10
    assert result["collection_name"] == techcorp_collection_name
    assert len(result["documents"]) == 10


def test_department_filter_returns_only_hr_documents(client, techcorp_collection_name):
    _load_techcorp_data(client, techcorp_collection_name)

    results = _search(
        client,
        techcorp_collection_name,
        query="onboarding process",
        filters={"department": "hr"},
    )

    assert results
    assert all(result["metadata"].get("department") == "hr" for result in results)


def test_combined_sales_confidential_filter_returns_only_matching_documents(client, techcorp_collection_name):
    _load_techcorp_data(client, techcorp_collection_name)

    results = _search(
        client,
        techcorp_collection_name,
        query="contract negotiation",
        filters={"department": "sales", "confidential": True},
    )

    assert len(results) == 2
    assert all(result["metadata"].get("department") == "sales" for result in results)
    assert all(result["metadata"].get("confidential") is True for result in results)


def test_security_tag_filter_returns_security_documents(client, techcorp_collection_name):
    _load_techcorp_data(client, techcorp_collection_name)

    results = _search(
        client,
        techcorp_collection_name,
        query="security breach response",
        filters={"tags": {"$contains": "security"}},
    )

    assert len(results) == 2
    assert all("security" in result["metadata"].get("tags", []) for result in results)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))