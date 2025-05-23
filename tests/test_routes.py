import pytest
from httpx import AsyncClient
from fastapi import status
from app.main import app


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "ok" or response.json()["status"] == "error"


@pytest.mark.asyncio
async def test_summarize_invalid_url():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {"youtube_url": "not_a_real_url"}
        response = await ac.post("/summarize/", json=payload)
    # You may want to adjust this depending on your validation
    assert response.status_code in (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# You can add more tests for /summarize/ and /question/ endpoints.
# For /question/, you may want to mock Redis and service layer dependencies.

# Example (pseudo-code, requires more setup):
# @pytest.mark.asyncio
# async def test_question_with_invalid_session():
#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         payload = {"session_id": "fake-session", "question": "What is this?"}
#         response = await ac.post("/question/", json=payload)
#     assert response.status_code == status.HTTP_404_NOT_FOUND
