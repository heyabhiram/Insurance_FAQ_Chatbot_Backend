import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.llm_service import STANDARD_FALLBACK_MESSAGE

client = TestClient(app)

def test_health_check_endpoint():
    """Test GET /health returns 200 and valid status dictionary."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Vehicle Insurance" in data["service"]
    assert data["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert data["chroma_db_status"] == "ready"
    assert data["faq_indexed_count"] >= 50


def test_get_all_faqs_endpoint():
    """Test GET /api/v1/faqs returns all indexed FAQs."""
    response = client.get("/api/v1/faqs")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 50
    assert len(data["faqs"]) == data["total"]
    first = data["faqs"][0]
    assert "id" in first
    assert "question" in first
    assert "answer" in first


def test_get_faqs_with_category_filter():
    """Test GET /api/v1/faqs?category=idv filters correctly."""
    response = client.get("/api/v1/faqs?category=idv")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
    assert data["category"] == "idv"
    for faq in data["faqs"]:
        assert faq["category"] == "idv"


def test_chat_endpoint_valid_question():
    """Test POST /api/v1/chat with a standard vehicle insurance query."""
    payload = {
        "message": "What is comprehensive car insurance?",
        "conversation_id": "test_session_001"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert data["intent"] in ["comprehensive_insurance", "policy_coverage"]
    assert len(data["sources"]) > 0
    assert data["confidence_score"] > 0.0


def test_chat_endpoint_idv_question():
    """Test POST /api/v1/chat with IDV question."""
    payload = {
        "message": "What is Insured Declared Value (IDV)?",
        "conversation_id": "test_session_idv"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "idv"
    assert any("idv" in s.lower() or "faq_005" in s for s in data["sources"])


def test_chat_contextual_followup():
    """Test multi-turn conversation maintains context for follow-up questions."""
    conv_id = "test_context_session"

    # Turn 1
    t1_response = client.post(
        "/api/v1/chat",
        json={"message": "What is IDV in car insurance?", "conversation_id": conv_id}
    )
    assert t1_response.status_code == 200

    # Turn 2: Follow-up referencing 'it' and 'premium'
    t2_response = client.post(
        "/api/v1/chat",
        json={"message": "How does it affect my premium?", "conversation_id": conv_id}
    )
    assert t2_response.status_code == 200
    t2_data = t2_response.json()
    # Should resolve to IDV affecting premium FAQ (faq_007)
    assert len(t2_data["sources"]) > 0
    assert "faq_007" in t2_data["sources"] or "faq_005" in t2_data["sources"]


def test_chat_unsupported_question_fallback():
    """Test that out-of-domain / unsupported questions receive standard fallback."""
    payload = {
        "message": "Can you give me the best recipe for baking chocolate cupcakes?",
        "conversation_id": "test_fallback_session"
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == STANDARD_FALLBACK_MESSAGE
    assert data["sources"] == []
    assert data["confidence_score"] == 0.0


def test_chat_empty_message_validation():
    """Test sending empty message is rejected with 422 or 400."""
    response = client.post("/api/v1/chat", json={"message": "", "conversation_id": "test"})
    assert response.status_code in [400, 422]


def test_chat_greetings_avoid_llm():
    """Test various greeting formats (Hello, Hai, Hiii, hi, heyy) return instant greeting and bypass LLM."""
    greetings = ["Hello", "Hai", "Hiii", "hi", "heyy", "Hey there!"]
    for greet in greetings:
        response = client.post(
            "/api/v1/chat",
            json={"message": greet, "conversation_id": "test_greeting_session"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "greeting"
        assert "Hello! I am your AI Vehicle Insurance Assistant" in data["answer"]
        assert data["sources"] == []
        assert data["confidence_score"] == 1.0


def test_chat_meta_history_questions_above():
    """Test retrieving questions asked N turns above (e.g. 'what was the question I asked 3 questions above')."""
    conv_id = "test_meta_history_session"
    client.post("/api/v1/chat", json={"message": "What is IDV in car insurance?", "conversation_id": conv_id})
    client.post("/api/v1/chat", json={"message": "What is NCB bonus?", "conversation_id": conv_id})
    client.post("/api/v1/chat", json={"message": "What is zero depreciation?", "conversation_id": conv_id})

    # Ask for 3 questions above (which was "What is IDV in car insurance?")
    resp_3 = client.post(
        "/api/v1/chat",
        json={"message": "what was the question I asked 3 questions above", "conversation_id": conv_id}
    )
    assert resp_3.status_code == 200
    data_3 = resp_3.json()
    assert data_3["intent"] == "meta_history"
    assert "What is IDV in car insurance?" in data_3["answer"]

    # Ask for previous/last question (which was "What is zero depreciation?")
    resp_1 = client.post(
        "/api/v1/chat",
        json={"message": "what was my last question", "conversation_id": conv_id}
    )
    assert resp_1.status_code == 200
    data_1 = resp_1.json()
    assert data_1["intent"] == "meta_history"
    assert "What is zero depreciation?" in data_1["answer"]


