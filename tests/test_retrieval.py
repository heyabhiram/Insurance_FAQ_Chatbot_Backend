import json
from pathlib import Path
import pytest
from app.config import settings
from app.services.nlp_service import nlp_service
from app.services.retrieval_service import retrieval_service

def test_faq_dataset_structure():
    """Verify FAQ JSON dataset exists, has at least 50 entries, and required keys."""
    assert settings.FAQ_DATA_PATH.exists(), f"FAQ dataset missing at {settings.FAQ_DATA_PATH}"

    with open(settings.FAQ_DATA_PATH, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    assert len(faqs) >= 50, f"Expected at least 50 FAQs, got {len(faqs)}"

    required_keys = {"id", "question", "answer", "category", "keywords", "source"}
    for idx, faq in enumerate(faqs):
        missing = required_keys - set(faq.keys())
        assert not missing, f"FAQ at index {idx} ({faq.get('id')}) is missing keys: {missing}"
        assert isinstance(faq["keywords"], list), f"Keywords must be a list in {faq.get('id')}"
        assert len(faq["question"].strip()) > 0
        assert len(faq["answer"].strip()) > 0


def test_nlp_intent_recognition():
    """Test rule-based intent recognition on representative queries."""
    test_cases = [
        ("What does comprehensive insurance cover?", "comprehensive_insurance"),
        ("Is third-party insurance compulsory in India?", "third_party_insurance"),
        ("How do I file an accidental damage claim?", "claim_process"),
        ("Where can I track my claim status?", "claim_status"),
        ("Can I pay my premium through UPI or credit card?", "premium_payment"),
        ("What is the grace period for policy renewal?", "policy_renewal"),
        ("How do I add a CNG kit endorsement to my policy?", "endorsement"),
        ("What is NCB and how much discount do I get?", "ncb"),
        ("How is IDV calculated for a 3-year-old car?", "idv"),
        ("What are the exclusions under car insurance?", "exclusions"),
        ("Who is eligible for compulsory personal accident cover?", "eligibility"),
        ("What does standalone own damage cover?", "policy_coverage"),
        ("Hello, who are you and how can you help?", "general_faq"),
        ("What is the recipe for chocolate cake?", "unknown"),
    ]

    for query, expected_intent in test_cases:
        detected = nlp_service.recognize_intent(query)
        assert detected == expected_intent, f"Query '{query}' expected intent '{expected_intent}', got '{detected}'"


def test_nlp_entity_extraction():
    """Test extracting entities like vehicle_type, policy_type, claim_number, policy_number."""
    query = "My bike claim CLM-98765 for comprehensive policy POL-123456 has IDV Rs. 75,000 renewal on 15/08/2025"
    entities = nlp_service.extract_entities(query)

    assert entities.get("vehicle_type") == "bike"
    assert entities.get("policy_type") == "comprehensive"
    assert "CLM-98765" in entities.get("claim_number", "")
    assert "POL-123456" in entities.get("policy_number", "")
    assert "75,000" in entities.get("idv_amount", "")
    assert "15/08/2025" in entities.get("renewal_date", "")


def test_nlp_no_invented_entities():
    """Verify that entity extraction does not hallucinate entities when absent."""
    query = "What is IDV?"
    entities = nlp_service.extract_entities(query)
    assert "claim_number" not in entities
    assert "policy_number" not in entities
    assert "vehicle_type" not in entities

