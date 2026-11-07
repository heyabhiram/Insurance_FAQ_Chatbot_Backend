import os
import logging
from typing import List, Dict, Any, Optional
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.config import settings

logger = logging.getLogger(__name__)

STANDARD_FALLBACK_MESSAGE = (
    "I couldn't establish a sufficiently relevant match for this query. This may be due to the way the question was phrased, missing context, or information outside the current knowledge scope. "
    "Please consider rephrasing the question or providing additional details."
)

SYSTEM_PROMPT = """You are an AI assistant specialized strictly in Vehicle Insurance FAQs.
Your role is to assist users by answering questions about vehicle insurance policies, claims, coverage, NCB, IDV, renewals, and endorsements.

STRICT OPERATING RULES:
1. Rely ONLY on the provided FAQ Context below. Do not use external or unverified information.
2. If the retrieved context is empty, irrelevant, or insufficient to answer the question accurately, reply EXACTLY with:
   "I couldn't establish a sufficiently relevant match for this query. This may be due to the way the question was phrased, missing context, or information outside the current knowledge scope. Please consider rephrasing the question or providing additional details."
3. NEVER fabricate or invent:
   - Policy coverage details not mentioned in context
   - Exact premium quotes, calculations, or discounts
   - Claim approval or claim status guarantees
   - Insurer-specific conditions or internal rules
   - Garage network names or addresses
4. Keep your explanations concise, professional, friendly, and easy for a beginner to understand.
5. Use previous conversation history only to understand follow-up references (e.g., pronouns like 'it', 'they', 'this cover').
"""

class LLMService:
    """
    Service encapsulating TCS GenAI Lab LLM integration and RAG response generation.
    """

    def __init__(self):
        self.api_key = settings.TCS_API_KEY or os.getenv("TCS_API_KEY", "")
        self.base_url = settings.TCS_BASE_URL
        self.model_name = settings.TCS_MODEL
        self.verify_ssl = settings.TCS_VERIFY_SSL
        self._llm = None

    def get_llm(self) -> Optional[ChatOpenAI]:
        """
        Initializes and returns the ChatOpenAI client using TCS GenAI Lab configuration.
        """
        # Re-check API key in case it was updated in environment
        current_key = settings.TCS_API_KEY or os.getenv("TCS_API_KEY", "")
        if not current_key or current_key.strip() == "":
            return None

        if self._llm is None or self.api_key != current_key:
            self.api_key = current_key
            client = httpx.Client(verify=self.verify_ssl, timeout=30.0)
            self._llm = ChatOpenAI(
                base_url=self.base_url,
                model=self.model_name,
                api_key=self.api_key,
                http_client=client,
                temperature=0.2,
                max_tokens=500
            )
        return self._llm

    def test_connection(self) -> Dict[str, Any]:
        """
        Tests connection to the TCS GenAI Lab LLM endpoint using `llm.invoke("Hi")`.
        """
        llm = self.get_llm()
        if not llm:
            return {
                "status": "unconfigured",
                "message": "TCS_API_KEY is not set. Configure it in .env to enable the LLM."
            }
        try:
            response = llm.invoke("Hi")
            return {
                "status": "connected",
                "model": self.model_name,
                "base_url": self.base_url,
                "response_sample": str(response.content)[:50]
            }
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def generate_answer(
        self,
        query: str,
        retrieved_faqs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generates a grounded, natural conversational response using retrieved FAQs and chat history.
        Gracefully handles LLM absence or failure by directly using retrieved factual FAQs.
        """
        # If no relevant FAQs were found, return the verified fallback
        if not retrieved_faqs:
            return STANDARD_FALLBACK_MESSAGE

        # Format retrieved context
        context_parts = []
        for i, faq in enumerate(retrieved_faqs, start=1):
            context_parts.append(
                f"[Source {faq.get('id', f'faq_{i}')}]\n"
                f"Question: {faq.get('question', '')}\n"
                f"Verified Answer: {faq.get('answer', '')}"
            )
        context_str = "\n\n".join(context_parts)

        llm = self.get_llm()

        # If LLM is not configured, fall back to direct verified answer from top FAQ
        if not llm:
            logger.info("LLM not configured (no TCS_API_KEY). Providing verified answer directly from knowledge base.")
            top_faq = retrieved_faqs[0]
            return top_faq.get("answer", STANDARD_FALLBACK_MESSAGE)

        # Build messages with conversation context
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Append limited recent conversation history for follow-up continuity (last 4 turns)
        if conversation_history:
            for turn in conversation_history[-4:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # Add the current user turn with RAG context
        user_prompt = (
            f"RELEVANT FAQ CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Generate a clear, accurate, and customer-friendly response based strictly on the context above."
        )
        messages.append(HumanMessage(content=user_prompt))

        try:
            response = llm.invoke(messages)
            answer_text = response.content
            if isinstance(answer_text, list):
                # Handle cases where response.content is a list of blocks
                answer_text = " ".join([str(b) for b in answer_text])
            return answer_text.strip()
        except Exception as e:
            logger.warning(
                f"LLM API invocation failed ({e}). Falling back to verified knowledge base answer."
            )
            # Graceful fallback: return top FAQ answer directly without fabricating
            top_faq = retrieved_faqs[0]
            return top_faq.get("answer", STANDARD_FALLBACK_MESSAGE)

# Singleton instance
llm_service = LLMService()

