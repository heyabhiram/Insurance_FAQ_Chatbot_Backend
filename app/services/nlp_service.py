import re
from typing import Dict, Any, List
from app.utils.preprocessing import normalize_text

class NLPService:
    """
    Lightweight rule-based NLP service for Intent Recognition and Entity Extraction.
    Designed to be transparent, extensible, and fast without unnecessary complex dependencies.
    """

    def __init__(self):
        # Specific intent order: more specific domains checked before general ones
        self.intent_order = [
            "greeting",
            "meta_history",
            "ncb",
            "idv",
            "endorsement",
            "claim_status",
            "claim_process",
            "exclusions",
            "eligibility",
            "third_party_insurance",
            "comprehensive_insurance",
            "policy_renewal",
            "premium_payment",
            "policy_coverage",
            "general_faq",
        ]

        self.intent_patterns = {
            "greeting": [
                r"^(?:h+[eia]+y*|h+a+i+|h+e+l+o+|hiya|hola|howdy|namaste|sup|yo|good\s+(?:morning|afternoon|evening|day)|greetings)(?:\s+(?:there|bot|assistant|drivesure|all|team|friend))?[\s!?.~]*$"
            ],
            "meta_history": [
                r"\b(?:what was|what were|show me|tell me|repeat)\b.*\b(?:question|query|queries)\b",
                r"\bwhat (?:did|have) i (?:just )?ask(?:ed)?\b",
                r"\bprevious question[s]?\b",
                r"\blast question[s]?\b",
                r"\b\d+\s*questions?\s*(?:above|ago|back|before)\b",
                r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*questions?\s*(?:above|ago|back|before)\b",
                r"\bwhat was my (?:\d+|first|second|third|fourth|fifth|last|previous) question\b",
            ],
            "ncb": [
                r"\bncb\b",
                r"\bno claim bonus\b",
                r"\bbonus discount\b",
                r"\bncb slab[s]?\b",
                r"\btransfer ncb\b",
                r"\bncb protect\b"
            ],
            "idv": [
                r"\bidv\b",
                r"\binsured declared value\b",
                r"\bvehicle value\b",
                r"\bmarket value\b",
                r"\bcalculate idv\b"
            ],
            "endorsement": [
                r"\bendorsement\b",
                r"\bendorse\b",
                r"\bchange name\b",
                r"\bupdate details\b",
                r"\btransfer ownership\b",
                r"\bcng kit\b",
                r"\blpg kit\b",
                r"\bhypothecation\b",
                r"\badd accessories\b",
                r"\bcorrection\b"
            ],
            "claim_status": [
                r"\bclaim status\b",
                r"\btrack(?:ing)? (?:my )?claim\b",
                r"\bstatus of (?:my )?claim\b",
                r"\bclaim progress\b",
                r"\bcheck claim\b",
                r"\bclaim delayed\b",
                r"\bclaim rejected\b"
            ],
            "claim_process": [
                r"\bfile\b.*\bclaim\b",
                r"\braise\b.*\bclaim\b",
                r"\bmake\b.*\bclaim\b",
                r"\bclaim process\b",
                r"\bclaim procedure\b",
                r"\bhow to claim\b",
                r"\bregister\b.*\bclaim\b",
                r"\baccident(?:al)?\b.*\bclaim\b",
                r"\btheft\b.*\bclaim\b",
                r"\bstolen\b",
                r"\bcashless claim\b",
                r"\breimbursement claim\b",
                r"\bclaim documents\b",
                r"\bsurveyor\b",
                r"\bnetwork garage\b",
                r"\bcashless garage\b"
            ],
            "exclusions": [
                r"\bexclusion[s]?\b",
                r"\bnot covered\b",
                r"\bdeductible[s]?\b",
                r"\bcompulsory deductible\b",
                r"\bvoluntary deductible\b",
                r"\bwear and tear\b",
                r"\bwithout licence\b",
                r"\bdrunk driving\b",
                r"\bmechanical breakdown\b"
            ],
            "eligibility": [
                r"\beligib(?:ility|le)\b",
                r"\bwho can buy\b",
                r"\bpre-owned\b",
                r"\bsecond[- ]hand\b",
                r"\bdocuments required to buy\b",
                r"\bpersonal accident\b",
                r"\bcpa cover\b"
            ],
            "third_party_insurance": [
                r"\bthird[-\s]?party\b",
                r"\btp insurance\b",
                r"\btp cover\b",
                r"\bliability cover\b",
                r"\bact only\b",
                r"\bmandatory insurance\b"
            ],
            "comprehensive_insurance": [
                r"\bcomprehensive\b",
                r"\bfull coverage\b",
                r"\ball round\b",
                r"\bpackage policy\b",
                r"\bcomprehensive vs third party\b"
            ],
            "policy_renewal": [
                r"\brenew(?:al|ing)?\b",
                r"\bexpired?\b",
                r"\blapse[d]?\b",
                r"\bbreak in insurance\b",
                r"\bgrace period\b"
            ],
            "premium_payment": [
                r"\bpremium\b",
                r"\bpayment\b",
                r"\bpay(?:ing)?\b",
                r"\bcost\b",
                r"\bprice\b",
                r"\bemi\b",
                r"\binstallments\b",
                r"\brefund\b"
            ],
            "policy_coverage": [
                r"\bcover(?:age)?\b",
                r"\bown damage\b",
                r"\bod cover\b",
                r"\badd[- ]?on\b",
                r"\bzero dep\b",
                r"\bnil dep\b",
                r"\bbumper to bumper\b",
                r"\broadside assistance\b",
                r"\brsa\b",
                r"\bengine protect\b",
                r"\breturn to invoice\b",
                r"\brti\b",
                r"\bflood damage\b",
                r"\bfire cover\b",
                r"\bkey replacement\b",
                r"\btyre protect\b",
                r"\bpolicy benefits\b"
            ],
            "general_faq": [
                r"\bhello\b",
                r"\bhi\b",
                r"\bhey\b",
                r"\bhelp\b",
                r"\bcontact\b",
                r"\bsupport\b",
                r"\bwhat can you do\b",
                r"\bwho are you\b"
            ]
        }

        # Entity extraction regex patterns
        self.entity_patterns = {
            "claim_number": [
                r"\b(?:clm|claim)[-_ ]?([a-zA-Z0-9]{4,12})\b",
                r"\bclaim\s*#?\s*([0-9]{4,12})\b"
            ],
            "policy_number": [
                r"\b(?:pol|policy)[-_ ]?([a-zA-Z0-9]{5,15})\b",
                r"\bpolicy\s*#?\s*([0-9]{5,15})\b"
            ],
            "idv_amount": [
                r"\b(?:rs\.?|inr|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:lakh|lakhs|crore|crores|k))?)\b",
                r"\b(\d+(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores))\b"
            ],
            "renewal_date": [
                r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
                r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
                r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{2,4})?)\b"
            ]
        }

        # Specific dictionary keywords for categoric entities
        self.vehicle_types = [
            "car", "bike", "two-wheeler", "two wheeler", "scooter",
            "four-wheeler", "four wheeler", "suv", "sedan", "hatchback",
            "motorcycle", "commercial vehicle", "truck", "ev", "electric vehicle"
        ]

        self.policy_types = [
            "comprehensive", "third-party", "third party", "standalone own damage",
            "own damage", "package policy", "zero depreciation", "bumper to bumper"
        ]

        self.coverage_types = [
            "own damage", "third party liability", "zero depreciation",
            "nil depreciation", "roadside assistance", "rsa",
            "engine protect", "return to invoice", "personal accident",
            "cpa", "key replacement", "tyre protect", "ncb protect"
        ]

    def recognize_intent(self, query: str) -> str:
        """
        Identifies the primary intent of the user's message using prioritized rule matching.
        """
        normalized_query = normalize_text(query)

        # Match against predefined intent patterns in priority order
        for intent in self.intent_order:
            patterns = self.intent_patterns.get(intent, [])
            for pattern in patterns:
                if re.search(pattern, normalized_query, re.IGNORECASE):
                    return intent

        return "unknown"

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extracts relevant vehicle insurance entities from the query.
        Returns only entities found in the query without fabricating values.
        """
        normalized_query = normalize_text(query)
        entities: Dict[str, Any] = {}

        # 1. Vehicle Type
        for v_type in self.vehicle_types:
            if re.search(rf"\b{re.escape(v_type)}\b", normalized_query):
                entities["vehicle_type"] = v_type
                break

        # 2. Policy Type
        for p_type in self.policy_types:
            if re.search(rf"\b{re.escape(p_type)}\b", normalized_query):
                entities["policy_type"] = p_type
                break

        # 3. Coverage Type
        for c_type in self.coverage_types:
            if re.search(rf"\b{re.escape(c_type)}\b", normalized_query):
                entities["coverage_type"] = c_type
                break

        # 4. Regex Pattern Entities
        # Claim Number
        for pattern in self.entity_patterns["claim_number"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["claim_number"] = match.group(0).strip()
                break

        # Policy Number
        for pattern in self.entity_patterns["policy_number"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["policy_number"] = match.group(0).strip()
                break

        # IDV Amount
        for pattern in self.entity_patterns["idv_amount"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["idv_amount"] = match.group(0).strip()
                break

        # Renewal Date
        for pattern in self.entity_patterns["renewal_date"]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities["renewal_date"] = match.group(0).strip()
                break

        # 5. Question Offset for Meta-History Queries (e.g. '3 questions above')
        word_to_num = {
            "one": 1, "first": 1, "1st": 1,
            "two": 2, "second": 2, "2nd": 2,
            "three": 3, "third": 3, "3rd": 3,
            "four": 4, "fourth": 4, "4th": 4,
            "five": 5, "fifth": 5, "5th": 5,
            "six": 6, "sixth": 6, "6th": 6,
            "seven": 7, "seventh": 7, "7th": 7,
            "eight": 8, "eighth": 8, "8th": 8,
            "nine": 9, "ninth": 9, "9th": 9,
            "ten": 10, "tenth": 10, "10th": 10,
            "last": 1, "previous": 1, "prior": 1,
        }
        offset_match = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|last|previous|prior)\s*questions?\s*(?:above|ago|back|before)\b",
            normalized_query
        )
        if not offset_match:
            offset_match = re.search(
                r"\b(?:what was|tell me) (?:the|my)?\s*(\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|last|previous)\s*question\b",
                normalized_query
            )

        if offset_match:
            raw_val = offset_match.group(1).lower()
            if raw_val.isdigit():
                entities["question_offset"] = int(raw_val)
            elif raw_val in word_to_num:
                entities["question_offset"] = word_to_num[raw_val]
        elif re.search(r"\b(?:previous|last|prior)\s*question\b", normalized_query):
            entities["question_offset"] = 1

        return entities

    def process(self, query: str) -> Dict[str, Any]:
        """
        Convenience method to execute intent recognition and entity extraction together.
        """
        return {
            "intent": self.recognize_intent(query),
            "entities": self.extract_entities(query),
        }

# Global singleton instance
nlp_service = NLPService()
