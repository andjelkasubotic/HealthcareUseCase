import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_and_get_draft(client) -> None:
    create_response = await client.post(
        "/drafts",
        json={"message": "I have chest pain for two days"},
    )
    assert create_response.status_code == 201

    created = create_response.json()
    assert created["input_message"] == "I have chest pain for two days"
    assert created["classifications"] == ["routine"]
    assert created["structured_fields"]["symptoms"] == "unspecified symptom"
    assert created["status"] == "ready"
    assert created["draft_response"]
    assert created["judge_result"]["approved"] is True
    assert created["needs_human_review"] is False

    draft_id = created["id"]
    get_response = await client.get(f"/drafts/{draft_id}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == draft_id


@pytest.mark.asyncio
async def test_get_missing_draft_returns_404(client) -> None:
    response = await client.get("/drafts/00000000-0000-0000-0000-000000000099")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_draft_rejects_empty_message(client) -> None:
    response = await client.post("/drafts", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_emergent_message_sets_abstraction_flag(client) -> None:
    response = await client.post(
        "/drafts",
        json={"message": "This is an emergency with severe pain"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["classifications"] == ["emergent"]
    assert body["abstraction_flag"] is True
