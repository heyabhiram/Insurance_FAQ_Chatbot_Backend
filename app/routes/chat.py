import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field

from app.services.nlp_service import nlp_service
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service, STANDARD_FALLBACK_MESSAGE
from app.services.history_service import history_service
from app.utils.preprocessing import normalize_text, is_greeting

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Request / Response Models ---

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User question or input message.")
    conversation_id: str = Field(default="default_session", description="Unique conversation thread identifier.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is comprehensive car insurance?",
                "conversation_id": "demo_001"
            }
        }
    }

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Conversational or verified FAQ answer.")
    intent: str = Field(..., description="Detected user intent.")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities from user input.")
    sources: List[str] = Field(default_factory=list, description="IDs of source FAQs used for the answer.")
    confidence_score: float = Field(..., description="Retrieval similarity confidence score between 0.0 and 1.0.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Comprehensive insurance provides coverage for damage to your vehicle...",
                "intent": "comprehensive_insurance",
                "entities": {"vehicle_type": "car", "policy_type": "comprehensive"},
                "sources": ["faq_001"],
                "confidence_score": 0.91
            }
        }
    }

class FAQItem(BaseModel):
    id: str
    question: str
    answer: str
    category: str
    keywords: List[str]
    source: str

class FAQListResponse(BaseModel):
    total: int
    category: Optional[str] = None
    faqs: List[FAQItem]


# --- Chat Endpoint ---

@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with the Vehicle Insurance Bot",
    description="Processes natural language vehicle insurance questions using NLP, RAG, and LLM.",
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    user_query = request.message.strip()
    conversation_id = request.conversation_id.strip() or "default_session"

    if not user_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty."
        )

    # 1. Normalize query
    normalized_query = normalize_text(user_query)

    # 2. Early Guard: Pure Greeting Detection (Zero LLM, Zero Vector Search)
    if is_greeting(user_query):
        greeting_answer = (
            "Hello! I am your AI Vehicle Insurance Assistant. "
            "You can ask me questions about comprehensive or third-party policies, "
            "IDV calculation, No Claim Bonus (NCB), claim procedures, deductibles, or policy renewals."
        )
        history_service.add_message(conversation_id, "user", user_query)
        history_service.add_message(conversation_id, "assistant", greeting_answer)
        return ChatResponse(
            answer=greeting_answer,
            intent="greeting",
            entities={},
            sources=[],
            confidence_score=1.0
        )

    # 3. NLP Processing: Intent Recognition & Entity Extraction
    nlp_result = nlp_service.process(normalized_query)
    intent = nlp_result["intent"]
    entities = nlp_result["entities"]

    # In case intent recognized greeting via NLP patterns
    if intent == "greeting":
        greeting_answer = (
            "Hello! I am your AI Vehicle Insurance Assistant. "
            "You can ask me questions about comprehensive or third-party policies, "
            "IDV calculation, No Claim Bonus (NCB), claim procedures, deductibles, or policy renewals."
        )
        history_service.add_message(conversation_id, "user", user_query)
        history_service.add_message(conversation_id, "assistant", greeting_answer)
        return ChatResponse(
            answer=greeting_answer,
            intent="greeting",
            entities=entities,
            sources=[],
            confidence_score=1.0
        )

    # 4. Early Guard: Meta-History Queries (e.g., 'what was the question I asked 3 questions above')
    if intent == "meta_history":
        offset = entities.get("question_offset", 1)
        # Fetch past questions asked by the user in this session (ordered recent to oldest)
        past_user_questions = history_service.get_past_user_questions(conversation_id, limit=20)

        # Filter out any pure greetings and meta-history queries so only substantive questions are counted
        substantive_questions = [
            q for q in past_user_questions
            if not is_greeting(q) and nlp_service.recognize_intent(q) != "meta_history"
        ]

        if not substantive_questions:
            answer = "You haven't asked any previous questions in this session yet. Feel free to ask anything about vehicle insurance!"
        elif offset <= len(substantive_questions):
            target_question = substantive_questions[offset - 1]
            if offset == 1:
                answer = f"Your previous question was:\n\n\"{target_question}\""
            else:
                answer = f"The question you asked {offset} questions ago was:\n\n\"{target_question}\""
        else:
            count = len(substantive_questions)
            questions_list = "\n".join(f"{i+1}. \"{q}\"" for i, q in enumerate(substantive_questions))
            answer = (
                f"You have only asked {count} question{'s' if count > 1 else ''} so far in this session:\n\n"
                f"{questions_list}"
            )

        history_service.add_message(conversation_id, "user", user_query)
        history_service.add_message(conversation_id, "assistant", answer)
        return ChatResponse(
            answer=answer,
            intent="meta_history",
            entities=entities,
            sources=[],
            confidence_score=1.0
        )

    # 5. Retrieve recent conversation history for domain context
    history = history_service.get_recent_history(conversation_id, limit=8)

    # 6. Contextual search query formulation:
    # If the user query is a short follow-up or contains pronouns ('it', 'this', 'that', 'they'),
    # enrich the search query using the previous user question to preserve contextual retrieval.
    search_query = user_query
    is_pronoun_followup = any(token in normalized_query.split() for token in ["it", "this", "that", "they", "its"])
    is_short_followup = len(normalized_query.split()) <= 4 and not is_greeting(user_query)

    if (is_pronoun_followup or is_short_followup) and history:
        last_user_turn = next(
            (turn["content"] for turn in reversed(history) if turn["role"] == "user" and not is_greeting(turn["content"])),
            None
        )
        if last_user_turn:
            search_query = f"{last_user_turn} {user_query}"

    # 7. Semantic FAQ Retrieval from ChromaDB
    retrieved_faqs = retrieval_service.retrieve(search_query, top_k=3)

    # If contextual query didn't return matches, retry with the direct query
    if not retrieved_faqs and search_query != user_query:
        retrieved_faqs = retrieval_service.retrieve(user_query, top_k=3)

    # 8. Evaluate Retrieval & Compute Confidence Score
    sources: List[str] = []
    confidence_score: float = 0.0

    if retrieved_faqs:
        sources = [faq["id"] for faq in retrieved_faqs]
        confidence_score = retrieved_faqs[0].get("similarity_score", 0.0)
    else:
        # Check if generic help query
        if intent == "general_faq" and any(w in normalized_query for w in ["help", "contact", "support", "who are you"]):
            answer = (
                "Hello! I am your AI Vehicle Insurance Assistant. "
                "You can ask me questions about comprehensive or third-party policies, "
                "IDV calculation, No Claim Bonus (NCB), claim procedures, deductibles, or policy renewals."
            )
            history_service.add_message(conversation_id, "user", user_query)
            history_service.add_message(conversation_id, "assistant", answer)
            return ChatResponse(
                answer=answer,
                intent=intent,
                entities=entities,
                sources=[],
                confidence_score=0.95
            )

    # 9. Generate Answer via TCS GenAI Lab LLM (or verified fallback)
    answer = llm_service.generate_answer(
        query=user_query,
        retrieved_faqs=retrieved_faqs,
        conversation_history=history
    )

    # 10. Save Conversation History for multi-turn context
    history_service.add_message(conversation_id, "user", user_query)
    history_service.add_message(conversation_id, "assistant", answer)

    return ChatResponse(
        answer=answer,
        intent=intent,
        entities=entities,
        sources=sources,
        confidence_score=confidence_score
    )


# --- FAQs Listing Endpoint ---

@router.get(
    "/faqs",
    response_model=FAQListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Vehicle Insurance FAQs",
    description="Retrieve all FAQs with optional category filtering (e.g., coverage, idv, ncb, claim_process).",
)
async def get_faqs(
    category: Optional[str] = Query(
        default=None,
        description="Filter by FAQ category (e.g., coverage, idv, ncb, claim_process, exclusions)"
    )
) -> FAQListResponse:
    faqs_data = retrieval_service.get_all_faqs(category=category)
    faq_items = [FAQItem(**item) for item in faqs_data]
    return FAQListResponse(
        total=len(faq_items),
        category=category,
        faqs=faq_items
    )

