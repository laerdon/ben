from typing import List, Dict, Set, Optional, Tuple, Any
import re
from dataclasses import dataclass
from enum import Enum, auto
from .llm import OllamaClient


class IntentType(Enum):
    """Enum of possible intent types."""

    GREETING = auto()  # General greeting or small talk
    QUESTION = auto()  # General information seeking
    RETRIEVAL = auto()  # Specific memory/knowledge retrieval
    MEMORY_GAIN = auto()  # Store new information
    MEMORY_LOSS = auto()  # Remove/forget information
    COMMAND = auto()  # System command or action request
    CLARIFICATION = auto()  # Asking for clarification
    OPINION = auto()  # Seeking opinion/evaluation
    CONTINUITY = auto()  # Continue previous conversation
    FEEDBACK = auto()  # Providing feedback
    UNKNOWN = auto()  # Intent not recognized


class EntityType(Enum):
    """Enum of possible entity types that can be extracted from messages."""

    DATE = auto()  # Date reference
    TOPIC = auto()  # Subject/topic
    KEYWORD = auto()  # Important keyword
    PERSON = auto()  # Person name
    PROJECT = auto()  # Project name
    ACTION = auto()  # Action to take
    SENTIMENT = auto()  # Positive/negative sentiment
    IMPORTANCE = auto()  # Importance indicator


@dataclass
class Entity:
    """Represents an entity extracted from a message."""

    type: EntityType
    value: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class IntentResult:
    """Holds the results of intent recognition."""

    primary_intent: IntentType
    secondary_intents: Set[IntentType]
    entities: List[Entity]
    confidence: float
    modified_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class IntentRecognizer:
    """Advanced intent recognition system for analyzing user messages."""

    def __init__(self, model: str = "llama3.1"):
        self.llm = OllamaClient(model=model)

        # Rule patterns for quick recognition
        self.patterns = {
            IntentType.GREETING: [
                r"^(hey|hi|hello|greetings|yo|howdy|what's up|sup)",
                r"^(good|happy) (morning|afternoon|evening|day)",
            ],
            IntentType.QUESTION: [
                r"\?$",
                r"^(what|when|where|who|whom|whose|which|why|how)",
                r"^(can|could|would|should|is|are|am|was|were|do|does|did) ",
            ],
            IntentType.RETRIEVAL: [
                r"(tell|show|find|search|look|get).*(about|for)",
                r"(what|when).*(happened|occurred|took place)",
                r"(remember|recall|retrieve)",
                r"(find|search|look).*(information|data)",
            ],
            IntentType.MEMORY_GAIN: [
                r"(important|remember|note|save|keep|store)",
                r"(this|that).*(matters|is important|is significant)",
                r"(don't forget|make sure|be sure)",
                r"(take note|write|jot|log)",
            ],
            IntentType.MEMORY_LOSS: [
                r"(forget|ignore|disregard|remove|delete)",
                r"(not important|doesn't matter|irrelevant)",
                r"(don't|do not).*(care|need|want)",
                r"(stop|quit).*(thinking|talking)",
            ],
            IntentType.COMMAND: [
                r"^(do|please|can you|could you)",
                r"^(list|show|find|get|create|make|update|change|modify|delete)",
                r"(run|execute|perform|implement)",
            ],
        }

        # Keywords associated with different intents
        self.keywords = {
            IntentType.RETRIEVAL: [
                "information",
                "details",
                "specifics",
                "data",
                "knowledge",
                "learn",
                "remember",
                "recall",
                "lookup",
                "check",
                "find out",
            ],
            IntentType.MEMORY_GAIN: [
                "important",
                "significant",
                "crucial",
                "essential",
                "key",
                "remember",
                "note",
                "save",
                "keep track",
                "don't forget",
                "highlight",
                "mark",
                "flag",
                "store",
                "archive",
                "preserve",
            ],
            IntentType.MEMORY_LOSS: [
                "forget",
                "ignore",
                "disregard",
                "unimportant",
                "irrelevant",
                "useless",
                "pointless",
                "meaningless",
                "trivial",
                "delete",
                "remove",
                "erase",
                "discard",
                "trash",
                "dump",
                "clear",
            ],
        }

    def recognize_intent(
        self, message: str, conversation_history: List[Dict] = None
    ) -> IntentResult:
        """
        Analyze a message to determine intent and extract entities.
        Uses a multi-stage approach combining rule-based patterns, keyword analysis,
        and LLM-based classification for more nuanced understanding.

        Args:
            message: The user message to analyze
            conversation_history: Optional list of previous messages for context

        Returns:
            IntentResult containing the recognized intent and entities
        """
        # Stage 1: Quick pattern matching for obvious intents
        quick_intents = self._apply_pattern_matching(message)

        # Stage 2: Extract potential entities
        entities = self._extract_entities(message)

        # Stage 3: Keyword analysis
        keyword_intents = self._analyze_keywords(message)

        # Stage 4: If pattern matching is inconclusive or complex, use LLM
        llm_analysis = None
        # Only use LLM if message is complex enough or initial analysis is ambiguous
        if len(message) > 15 or len(quick_intents) <= 1:
            llm_analysis = self._analyze_with_llm(
                message, conversation_history, quick_intents
            )

        # Stage 5: Intent resolution - combine and prioritize results
        final_result = self._resolve_intents(
            message, quick_intents, keyword_intents, llm_analysis, entities
        )

        return final_result

    def _apply_pattern_matching(self, message: str) -> Dict[IntentType, float]:
        """
        Apply regex pattern matching to quickly identify obvious intents.
        Returns a dict mapping intent types to confidence scores.
        """
        results = {}
        cleaned_message = message.lower().strip()

        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, cleaned_message, re.IGNORECASE):
                    # Increase confidence if multiple patterns match
                    current_confidence = results.get(intent_type, 0.0)
                    results[intent_type] = min(
                        current_confidence + 0.25, 0.9
                    )  # Cap at 0.9

        # Default to UNKNOWN if nothing matches
        if not results:
            results[IntentType.UNKNOWN] = 0.3

        return results

    def _extract_entities(self, message: str) -> List[Entity]:
        """
        Extract potential entities from the message using rule-based patterns.
        More sophisticated entity extraction would use NER models.
        """
        entities = []

        # Simple date extraction
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # M/D/Y
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}(st|nd|rd|th)?,? \d{2,4}\b",  # Month Day, Year
        ]

        for pattern in date_patterns:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                entities.append(
                    Entity(
                        type=EntityType.DATE,
                        value=match.group(0),
                        confidence=0.8,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )

        # Importance indicators
        importance_patterns = [
            r"\b(important|critical|crucial|essential|key|significant|vital|major)\b",
            r"\b(high|top)(-|\s)priority\b",
        ]

        for pattern in importance_patterns:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                entities.append(
                    Entity(
                        type=EntityType.IMPORTANCE,
                        value=match.group(0),
                        confidence=0.7,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )

        # Sentiment extraction (very basic)
        positive_terms = r"\b(good|great|excellent|amazing|wonderful|positive|love|like|enjoy|happy|glad)\b"
        negative_terms = r"\b(bad|terrible|awful|horrible|negative|hate|dislike|sad|upset|disappointed)\b"

        for match in re.finditer(positive_terms, message, re.IGNORECASE):
            entities.append(
                Entity(
                    type=EntityType.SENTIMENT,
                    value="positive",
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        for match in re.finditer(negative_terms, message, re.IGNORECASE):
            entities.append(
                Entity(
                    type=EntityType.SENTIMENT,
                    value="negative",
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
            )

        return entities

    def _analyze_keywords(self, message: str) -> Dict[IntentType, float]:
        """Analyze message for keywords associated with different intents."""
        results = {}
        message_lower = message.lower()

        for intent_type, keywords in self.keywords.items():
            matches = 0
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    matches += 1

            if matches > 0:
                confidence = min(
                    0.3 + (matches * 0.1), 0.8
                )  # Scale with matches but cap at 0.8
                results[intent_type] = confidence

        return results

    def _analyze_with_llm(
        self,
        message: str,
        conversation_history: List[Dict] = None,
        initial_intents: Dict[IntentType, float] = None,
    ) -> Dict[IntentType, float]:
        """Use the LLM for deeper intent analysis, with conversation context."""
        # Prepare conversation context
        context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_messages = conversation_history[-3:]  # Last 3 messages
            context = "Recent conversation:\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"

        # Include initial pattern-matched intents if available
        initial_analysis = ""
        if initial_intents:
            initial_analysis = "Initial analysis detected these potential intents:\n"
            for intent, conf in initial_intents.items():
                if conf > 0.5:  # Only include high-confidence initial results
                    initial_analysis += f"- {intent.name} (confidence: {conf:.2f})\n"

        # Build the prompt
        prompt = f"""
        {context}
        
        {initial_analysis}
        
        Analyze this user message: "{message}"
        
        Determine the user's intent from these categories:
        - GREETING: General greeting or small talk
        - QUESTION: General information seeking
        - RETRIEVAL: Specific memory/knowledge retrieval
        - MEMORY_GAIN: Store new information as important
        - MEMORY_LOSS: Remove/forget information
        - COMMAND: System command or action request
        - CLARIFICATION: Asking for clarification
        - OPINION: Seeking opinion/evaluation
        - CONTINUITY: Continue previous conversation
        - FEEDBACK: Providing feedback
        - UNKNOWN: Intent not recognized
        
        For each potential intent, assign a confidence score between 0 and 1.
        Format your response as:
        INTENT: intent_name, confidence_score
        
        You may include up to 3 intents, ordered by confidence.
        """

        response = self.llm._generate(prompt)

        # Parse the response
        results = {}
        for line in response.strip().split("\n"):
            if line.startswith("INTENT:"):
                parts = line.replace("INTENT:", "").strip().split(",")
                if len(parts) >= 2:
                    intent_name = parts[0].strip().upper()
                    try:
                        confidence = float(parts[1].strip())
                        # Map string intent name to enum
                        try:
                            intent_type = IntentType[intent_name]
                            results[intent_type] = min(
                                confidence, 1.0
                            )  # Ensure not > 1
                        except KeyError:
                            # Intent name not recognized, skip
                            pass
                    except ValueError:
                        # Couldn't parse confidence, skip
                        pass

        return results

    def _resolve_intents(
        self,
        message: str,
        pattern_intents: Dict[IntentType, float],
        keyword_intents: Dict[IntentType, float],
        llm_intents: Optional[Dict[IntentType, float]],
        entities: List[Entity],
    ) -> IntentResult:
        """
        Resolve different intent signals into a final decision with confidence scores.
        Prioritizes LLM analysis for complex queries, but considers all signals.
        """
        combined_intents = {}

        # Combine all intent sources with different weights
        for intent, conf in pattern_intents.items():
            combined_intents[intent] = combined_intents.get(intent, 0) + (
                conf * 0.3
            )  # 30% weight to patterns

        for intent, conf in keyword_intents.items():
            combined_intents[intent] = combined_intents.get(intent, 0) + (
                conf * 0.2
            )  # 20% weight to keywords

        if llm_intents:
            for intent, conf in llm_intents.items():
                combined_intents[intent] = combined_intents.get(intent, 0) + (
                    conf * 0.5
                )  # 50% weight to LLM

        # Normalize confidence scores
        total_confidence = sum(combined_intents.values())
        if total_confidence > 0:
            for intent in combined_intents:
                combined_intents[intent] /= total_confidence

        # Determine primary and secondary intents
        sorted_intents = sorted(
            combined_intents.items(), key=lambda x: x[1], reverse=True
        )

        primary_intent = IntentType.UNKNOWN
        primary_confidence = 0.0
        secondary_intents = set()

        if sorted_intents:
            primary_intent, primary_confidence = sorted_intents[0]

            # Add secondary intents if confidence is substantial (>0.2)
            for intent, conf in sorted_intents[1:]:
                if conf > 0.2:
                    secondary_intents.add(intent)

        # Special case: if GREETING is secondary with decent confidence and another intent is primary,
        # include it to support natural conversation flow
        if IntentType.GREETING not in (primary_intent, *secondary_intents):
            greeting_conf = combined_intents.get(IntentType.GREETING, 0)
            if greeting_conf > 0.15:
                secondary_intents.add(IntentType.GREETING)

        # Construct metadata with confidence breakdown for debugging
        metadata = {
            "confidence_breakdown": {
                intent.name: confidence
                for intent, confidence in combined_intents.items()
                if confidence > 0.1
            }
        }

        return IntentResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            entities=entities,
            confidence=primary_confidence,
            modified_message=message,  # Could be modified if needed
            metadata=metadata,
        )
