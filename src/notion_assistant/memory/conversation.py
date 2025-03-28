from typing import List, Dict, Optional, Tuple, Set, Callable
import json
import random
from datetime import datetime
from .manager import MemoryManager
from .insights import InsightGenerator
from .llm import OllamaClient
from .intent import IntentRecognizer, IntentType
import requests


class ConversationManager:
    def __init__(self, model: str = "llama3.1", debug: bool = False):
        self.memory_manager = MemoryManager()
        self.insight_generator = InsightGenerator()
        self.llm = OllamaClient(model=model)
        self.intent_recognizer = IntentRecognizer(model=model)
        self.conversation_history = []
        self.debug = debug

    def chat(
        self, user_message: str, stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Process a user message and generate a response using memory and insights.

        Args:
            user_message: The user's message to process
            stream_callback: Optional callback function to receive streaming output chunks

        Returns:
            The complete response
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        try:
            # Recognize intent using the new intent recognition system
            intent_result = self.intent_recognizer.recognize_intent(
                user_message, self.conversation_history
            )

            # Map recognized intents to behaviors
            behaviors = self._map_intents_to_behaviors(intent_result)

            # Apply memory operations based on behaviors
            if "memory_loss" in behaviors:
                self._apply_memory_loss(user_message)

            # Search memory for relevant entries (if retrieval is needed)
            memory_results = []
            if "retrieval" in behaviors:
                memory_results = self.memory_manager.search(user_message, top_k=3)

            # Load latest insights
            insights = self.insight_generator.load_latest_insights()

            # Build context for LLM
            context = self._build_context(user_message, memory_results, insights)

            # Generate response - use streaming if callback is provided
            if stream_callback:
                response = self._generate_response_stream(
                    context, behaviors, stream_callback
                )
            else:
                response = self._generate_response(context, behaviors)

            # Apply memory gain if needed
            if "memory_gain" in behaviors:
                self._apply_memory_gain(user_message, response)

            # Add behavior and intent information to response for debugging
            debug_info = ""
            if self.debug:
                # Add behaviors
                debug_info += f"\n\n[behaviors: {', '.join(behaviors)}]"

                # Add intent information
                debug_info += f"\n[primary intent: {intent_result.primary_intent.name} ({intent_result.confidence:.2f})]"
                if intent_result.secondary_intents:
                    secondary = ", ".join(
                        [i.name for i in intent_result.secondary_intents]
                    )
                    debug_info += f"\n[secondary intents: {secondary}]"

                # Add confidence breakdown if available
                if (
                    intent_result.metadata
                    and "confidence_breakdown" in intent_result.metadata
                ):
                    breakdown = ", ".join(
                        [
                            f"{intent}: {conf:.2f}"
                            for intent, conf in intent_result.metadata[
                                "confidence_breakdown"
                            ].items()
                        ]
                    )
                    debug_info += f"\n[confidence: {breakdown}]"

            full_response = response + debug_info

            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": response})

            return full_response
        except requests.exceptions.ConnectionError:
            error_msg = "sorry, i can't think right now. make sure ollama is running (http://localhost:11434)"
            self.conversation_history.append(
                {"role": "assistant", "content": error_msg}
            )
            return error_msg
        except Exception as e:
            error_msg = f"sorry, i'm having trouble thinking right now. error: {str(e)}"
            self.conversation_history.append(
                {"role": "assistant", "content": error_msg}
            )
            return error_msg

    def _map_intents_to_behaviors(self, intent_result) -> Set[str]:
        """Map the recognized intents to behavior flags."""
        behaviors = {"default"}

        # Map primary intent
        if intent_result.primary_intent == IntentType.RETRIEVAL:
            behaviors.add("retrieval")
        elif intent_result.primary_intent == IntentType.MEMORY_GAIN:
            behaviors.add("memory_gain")
        elif intent_result.primary_intent == IntentType.MEMORY_LOSS:
            behaviors.add("memory_loss")
        elif intent_result.primary_intent == IntentType.QUESTION:
            behaviors.add("retrieval")  # Questions typically need retrieval

        # Consider secondary intents too
        for intent in intent_result.secondary_intents:
            if intent == IntentType.RETRIEVAL:
                behaviors.add("retrieval")
            elif intent == IntentType.MEMORY_GAIN:
                behaviors.add("memory_gain")
            elif intent == IntentType.MEMORY_LOSS:
                behaviors.add("memory_loss")

        # Consider entities
        for entity in intent_result.entities:
            if entity.type.name == "IMPORTANCE" and entity.confidence > 0.6:
                behaviors.add("memory_gain")  # Important things should be remembered

        return behaviors

    def _evaluate_interaction_purpose(self, message: str) -> Tuple[Set[str], str]:
        """
        DEPRECATED: Use intent_recognizer instead.
        This method is kept for backward compatibility.
        """
        # Start with default behavior
        behaviors = {"default"}
        modified_message = message

        # Simple heuristics for behavior determination
        # For a more robust solution, we could use the LLM to classify the message

        # Check for retrieval behavior (questions, seeking information)
        retrieval_indicators = [
            "?",
            "what",
            "when",
            "where",
            "who",
            "how",
            "why",
            "tell me",
            "explain",
            "describe",
            "show",
            "find",
            "search",
            "looking for",
            "remember",
            "recall",
        ]
        if any(indicator in message.lower() for indicator in retrieval_indicators):
            behaviors.add("retrieval")

        # Check for memory loss behavior
        memory_loss_indicators = [
            "forget",
            "don't care",
            "not important",
            "ignore",
            "disregard",
            "don't remember",
            "not interested",
            "nevermind",
            "not relevant",
        ]
        if any(indicator in message.lower() for indicator in memory_loss_indicators):
            behaviors.add("memory_loss")

        # Check for memory gain behavior
        memory_gain_indicators = [
            "important",
            "remember this",
            "note this",
            "interesting",
            "significant",
            "crucial",
            "essential",
            "key",
            "valuable",
            "take note",
            "this matters",
        ]
        if any(indicator in message.lower() for indicator in memory_gain_indicators):
            behaviors.add("memory_gain")

        # Prompt the LLM for complex cases if the message is long enough and no clear behavior
        if len(message) > 20 and len(behaviors) <= 1:
            behaviors = behaviors.union(self._get_llm_behavior_evaluation(message))

        return behaviors, modified_message

    def _get_llm_behavior_evaluation(self, message: str) -> Set[str]:
        """
        DEPRECATED: Use intent_recognizer instead.
        This method is kept for backward compatibility.
        """
        prompt = f"""
        Analyze this user message and determine which behaviors should be applied:
        
        Message: "{message}"
        
        Possible behaviors:
        - retrieval: Does the user need information from memory?
        - memory_loss: Is the user indicating something should be forgotten/removed?
        - memory_gain: Is the user indicating something important that should be remembered?
        
        Return only the behavior names that apply, separated by commas.
        """

        result = self.llm._generate(prompt).lower().strip()
        behaviors = {
            b.strip()
            for b in result.split(",")
            if b.strip() in ["retrieval", "memory_loss", "memory_gain"]
        }
        return behaviors

    def _apply_memory_loss(self, message: str):
        """Apply memory loss by removing tokens from relevant entries."""
        # Find entries that might be relevant to what should be forgotten
        relevant_entries = self.memory_manager.search(message, top_k=2)

        for result in relevant_entries:
            # Skip if no ID available (should not happen now that we include IDs)
            if not result.entry.id:
                continue

            # Split the text into tokens (simple word-based approach)
            tokens = result.entry.raw_text.split()

            if len(tokens) <= 5:  # Don't modify very short entries
                continue

            # Remove a random selection of words (10-20% of tokens)
            num_to_remove = max(1, int(random.uniform(0.2, 0.3) * len(tokens)))
            indices_to_remove = random.sample(range(len(tokens)), num_to_remove)

            # Create modified text
            modified_tokens = [
                t for i, t in enumerate(tokens) if i not in indices_to_remove
            ]
            modified_text = " ".join(modified_tokens)

            # Update the entry in ChromaDB using the ID directly from the search result
            success = self.memory_manager.update_entry(result.entry.id, modified_text)

            if self.debug:
                if success:
                    print(f"Applied memory loss to entry: {result.entry.id}")
                else:
                    print(f"Failed to apply memory loss to entry: {result.entry.id}")

    def _apply_memory_gain(self, user_message: str, response: str):
        """Add important information to ChromaDB for today's date."""
        # Combine the user message and response as the content to remember
        content_to_remember = f"User: {user_message}\nResponse: {response}"

        # Add to ChromaDB with today's date
        today = datetime.now().strftime("%Y-%m-%d")

        # Add a new entry for today
        entry_id = self.memory_manager.add_entry_for_date(today, content_to_remember)

        if self.debug and entry_id:
            print(f"Added memory gain entry with ID: {entry_id}")
        elif self.debug:
            print("Failed to add memory gain entry")

    def _build_context(self, query, memory_results, insights):
        """Build context from memory and insights."""
        context = ""
        has_context = False

        # Add memory entries if available
        if memory_results:
            has_context = True
            context += "relevant log entries:\n"
            for i, result in enumerate(memory_results, 1):
                context += f"entry {i} ({result.entry.date.strftime('%Y-%m-%d')}):\n"
                # Limit text length to avoid token overload
                preview = result.entry.raw_text[:500]
                if len(result.entry.raw_text) > 500:
                    preview += "..."
                context += f"{preview}\n\n"

        # Add insights if available and not an error
        if insights and "error" not in insights and "windows" in insights:
            if insights["windows"]:
                has_context = True
                context += ""
                # Just use the most recent window
                window = insights["windows"][0]

                if "insights" in window and window["insights"]:
                    context += "key insights:\n"
                    for insight in window["insights"][:3]:  # Limit to top 3
                        context += f"- {insight}\n"

                if "themes" in window and window["themes"]:
                    context += "\nthemes:\n"
                    for theme in window["themes"][:3]:  # Limit to top 3
                        context += f"- {theme}\n"

        if not has_context:
            context += "i don't have any data in my memory yet. you can use the 'rebuild database from notion' option to load your entries."

        return context

    def _generate_response(self, context: str, behaviors: Set[str]) -> str:
        """Generate a response using the LLM."""
        # Build the conversation history for context
        history_text = ""
        if self.conversation_history:  # Include all conversation history
            # Get the last 5 messages at most (or all if less than 5)
            recent_messages = (
                self.conversation_history[-5:]
                if len(self.conversation_history) > 5
                else self.conversation_history
            )
            for msg in recent_messages:
                role = "you" if msg["role"] == "user" else "ben"
                history_text += f"{role}: {msg['content']}\n"

        behavior_guidance = ""
        if "memory_loss" in behaviors:
            behavior_guidance += "the user seems to want to forget or disregard something. acknowledge this appropriately.\n"
        if "memory_gain" in behaviors:
            behavior_guidance += "the user mentioned something important. acknowledge the importance of what they said.\n"

        prompt = f"""you are ben, a helpful and casual ai assistant that helps users understand their projects and notes.
you speak in lowercase only and have a laid-back style.

### INSTRUCTIONS ###
1. Only use "hey there" or "hi" in the very first message
2. For all follow-up messages, respond directly without any greeting phrases
3. Keep responses friendly, casual, and concise
4. Don't mention "memory", "logs", or "entries" - incorporate information naturally
5. All responses must be in lowercase only
{behavior_guidance}

conversation history:
{history_text}

context information:
{context}

respond to the user's most recent message in a conversational way that continues the existing conversation. 
be helpful and informative.
"""

        # Print prompt for debugging if enabled
        if self.debug:
            print("\n--- DEBUG: PROMPT SENT TO MODEL ---")
            print(prompt)
            print("--- END DEBUG PROMPT ---\n")

        response = self.llm._generate(prompt)

        # Print response for debugging if enabled
        if self.debug:
            print("\n--- DEBUG: RAW MODEL RESPONSE ---")
            print(response)
            print("--- END DEBUG RESPONSE ---\n")

        # Ensure the response is lowercase
        return response.strip().lower()

    def _generate_response_stream(
        self, context: str, behaviors: Set[str], stream_callback: Callable[[str], None]
    ) -> str:
        """Generate a response using the LLM and stream it to the callback."""
        # Build the conversation history for context
        history_text = ""
        if self.conversation_history:  # Include all conversation history
            # Get the last 5 messages at most (or all if less than 5)
            recent_messages = (
                self.conversation_history[-5:]
                if len(self.conversation_history) > 5
                else self.conversation_history
            )
            for msg in recent_messages:
                role = "you" if msg["role"] == "user" else "ben"
                history_text += f"{role}: {msg['content']}\n"

        behavior_guidance = ""
        if "memory_loss" in behaviors:
            behavior_guidance += "the user seems to want to forget or disregard something. acknowledge this appropriately.\n"
        if "memory_gain" in behaviors:
            behavior_guidance += "the user mentioned something important. acknowledge the importance of what they said.\n"

        prompt = f"""you are ben, a helpful and casual ai assistant that helps users understand their projects and notes.
you speak in lowercase only and have a laid-back style.

### INSTRUCTIONS ###
1. Only use "hey there" or "hi" in the very first message
2. For all follow-up messages, respond directly without any greeting phrases
3. Keep responses friendly, casual, and concise
4. Don't mention "memory", "logs", or "entries" - incorporate information naturally
5. All responses must be in lowercase only
{behavior_guidance}

conversation history:
{history_text}

context information:
{context}

respond to the user's most recent message in a conversational way that continues the existing conversation. 
be helpful and informative.
"""

        # Print prompt for debugging if enabled
        if self.debug:
            print("\n--- DEBUG: PROMPT SENT TO MODEL ---")
            print(prompt)
            print("--- END DEBUG PROMPT ---\n")

        # Use the streaming version of generate
        response = self.llm._generate_stream(prompt, callback=stream_callback)

        # Print response for debugging if enabled
        if self.debug:
            print("\n--- DEBUG: RAW MODEL RESPONSE ---")
            print(response)
            print("--- END DEBUG RESPONSE ---\n")

        # Ensure the response is lowercase
        return response.strip().lower()

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
