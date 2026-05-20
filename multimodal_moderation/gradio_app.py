"""
Gradio Chat Interface for ACME Customer Service Training

This module provides an interactive chat interface where users practice being customer service agents. 
The AI plays the role of a customer with complaints, and all service agent messages/media are moderated before being sent to the AI.

DATA FLOW:
1. Service agent (user) sends message/media → Gradio interface
2. Content sent to FastAPI moderation service (via HTTP)
3. If safe → Content sent to Gemini AI (which plays customer role)
4. AI customer response returned to user

KEY COMPONENTS:
- check_content_safety(): Calls FastAPI backend to moderate content
- ChatSessionWithTracing: Manages conversation state and tracing
- create_chat_interface(): Builds the Gradio UI
"""

import os
import requests
import gradio as gr
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Any
from pydantic_ai.messages import BinaryContent
import logging

from multimodal_moderation.env import USER_API_KEY, API_BASE_URL
from multimodal_moderation.tracing import setup_tracing, get_tracer, add_media_to_span
from multimodal_moderation.agents.customer_agent import customer_agent
from multimodal_moderation.utils import detect_file_type
from opentelemetry import trace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up tracing for observability with Phoenix
setup_tracing()
tracer = get_tracer(__name__)

# Constants
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Moderation Configuration
# Maps content types to their FastAPI backend endpoints and safety flags.
# This configuration determines which API endpoint to call for each content type
# and which flags indicate unsafe content.
MODERATION_CONFIG = {
    "text": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_text",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
    "image": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_image_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "video": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_video_file",
        "unsafe_flags": ["contains_pii", "is_disturbing", "is_low_quality"],
    },
    "audio": {
        "endpoint": f"{API_BASE_URL}/api/v1/moderate_audio_file",
        "unsafe_flags": ["is_unfriendly", "is_unprofessional", "contains_pii"],
    },
}


def _call_text_moderation(text: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    """
    Call the FastAPI backend to moderate text content.

    Makes an HTTP POST request to the text moderation endpoint.

    Returns: (result_dict, feedback, content_type, mime_type)
    Raises: RuntimeError if the moderation service is unavailable
    """
    content_type = "text"
    mime_type = "text/plain"

    config = MODERATION_CONFIG[content_type]

    # BACKEND CALL: HTTP POST to FastAPI moderation service
    response = requests.post(
        config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, json={"text": text}
    )

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    # Add input to tracing span for observability
    span.set_attributes(
        {
            "input.text.content": text,
            "input.text.length": len(text),
        }
    )

    # Parse the moderation result from the backend
    result = response.json()
    feedback = result["rationale"]

    return result, feedback, content_type, mime_type


def _call_media_moderation(media: str, span: trace.Span) -> Tuple[dict[str, Any], str, str, str]:
    """
    Call the FastAPI backend to moderate media files (image/video/audio).

    Detects file type, validates size, and sends to appropriate endpoint.

    Returns: (result_dict, feedback, content_type, mime_type)
    Raises: ValueError if file is too large, RuntimeError if service unavailable
    """
    # Detect MIME type (e.g., "image/png", "video/mp4", "audio/mpeg")
    mime_type = detect_file_type(media, context=media)

    # Validate file size to prevent large uploads
    file_size = os.path.getsize(media)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(f"File too large: {size_mb:.1f}MB. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    # Extract content type from MIME type (e.g., "image" from "image/png")
    content_type = mime_type.split("/")[0]
    config = MODERATION_CONFIG[content_type]

    # BACKEND CALL: HTTP POST with file upload to FastAPI moderation service
    with open(media, "rb") as f:
        response = requests.post(
            config["endpoint"], headers={"Authorization": f"Bearer {USER_API_KEY}"}, files={"file": f}
        )

    # Add media metadata to tracing span for Phoenix visualization
    add_media_to_span(span, media, f"{content_type}_moderation", 0)

    if not response.ok:
        raise RuntimeError(f"Moderation service unavailable. Please try again later. {response.text}")

    # Parse the moderation result from the backend
    result = response.json()
    feedback = result["rationale"]

    # Special case for audio: include transcription in feedback
    if content_type == "audio" and "transcription" in result:
        feedback = f"Transcription: \"{result['transcription']}\"\n\n{feedback}"

    return result, feedback, content_type, mime_type


def _emit_feedback_span(feedback: str, *, flagged: bool, content_type: str | None = None) -> None:
    """
    Emit a child "feedback" span recording what the trainee saw this turn.

    Created on every terminal path of a chat turn (flagged or safe) so the
    Phoenix trace always has a feedback record, not only when content is blocked.
    Must be called inside an active "chat_turn" span so it nests correctly.
    """
    with tracer.start_as_current_span("feedback") as feedback_span:
        feedback_span.set_attribute("feedback", feedback)
        feedback_span.set_attribute("flagged", flagged)
        if content_type:
            feedback_span.set_attribute("content_type", content_type)


def check_content_safety(*, text: str | None = None, media: str | None = None) -> Tuple[bool, str, str]:
    """
    Check if content is safe by calling the moderation backend.

    This is the main entry point for all content moderation. 
    It routes to either text or media moderation, then checks if any safety flags are set.

    Args:
        text: Text content to moderate (mutually exclusive with media)
        media: Path to media file to moderate (mutually exclusive with text)

    Returns:
        Tuple of (is_safe, feedback_message, mime_type)
    """
    # Create a tracing span for this moderation check
    with tracer.start_as_current_span("moderate_text") as span:

        # Route to the appropriate moderation function
        if text is not None:
            result, feedback, content_type, mime_type = _call_text_moderation(text, span)

        elif media is not None:
            result, feedback, content_type, mime_type = _call_media_moderation(media, span)

        else:
            raise ValueError("Must provide exactly one of text or media")

        # Add moderation results to tracing span
        span.set_attributes({f"output.{k}": v for k, v in result.items()})

        # Update span name now that we know the content type
        span.update_name(f"moderate_{content_type}")

    # Check if any unsafe flags were set by the moderation service
    config = MODERATION_CONFIG[content_type]
    for flag in config["unsafe_flags"]:
        if result[flag]:
            # Content is unsafe - return False with feedback
            return False, f"Content flagged: {feedback}", mime_type

    # Content is safe - return True with feedback
    return True, feedback, mime_type


BLOCKED_RESPONSE = "[This content was flagged by moderation and not sent to the AI. Please try again.]"


@dataclass
class _ModerationOutcome:
    """Result of moderating one chat turn's user input.

    If ``blocked_feedback`` is set, the turn must be blocked and no agent call
    should be made. Otherwise ``prompt_parts`` is the assembled prompt to forward
    to the customer agent and ``safety_message`` is the last moderation rationale.
    """

    prompt_parts: List[str | BinaryContent] = field(default_factory=list)
    safety_message: str = ""
    blocked_feedback: str | None = None
    blocked_content_type: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked_feedback is not None


def _moderate_turn_content(message: dict) -> _ModerationOutcome:
    """Moderate every text/file in ``message``; return the assembled prompt or a block reason.

    Stops at the first flagged item (text is checked before files) so the trainee
    gets feedback on the first violation rather than the last.
    """
    outcome = _ModerationOutcome(
        prompt_parts=["This is the next message from the support agent:"]
    )

    text = message.get("text")
    if text:
        is_safe, outcome.safety_message, _ = check_content_safety(text=text)
        if not is_safe:
            outcome.blocked_feedback = f"⚠️ Content flagged: {outcome.safety_message}"
            outcome.blocked_content_type = "text"
            return outcome
        outcome.prompt_parts.append(text)

    for file_path in message.get("files") or []:
        try:
            is_safe, outcome.safety_message, mime_type = check_content_safety(media=file_path)
        except ValueError as e:
            # File-too-large or similar input errors surface directly to the UI
            raise gr.Error(str(e))

        if not is_safe:
            outcome.blocked_feedback = f"⚠️ Content flagged: {outcome.safety_message}"
            # Derive content_type from mime_type (e.g. "image" from "image/png")
            outcome.blocked_content_type = mime_type.split("/")[0] if mime_type else None
            return outcome

        with open(file_path, "rb") as f:
            file_bytes = f.read()
        outcome.prompt_parts.append(BinaryContent(data=file_bytes, media_type=mime_type))

    return outcome


async def _run_customer_agent(
    prompt_parts: List[str | BinaryContent], past_messages: List
) -> Tuple[str, List]:
    """Send moderated content to the customer-role agent inside its own span."""
    try:
        with tracer.start_as_current_span("llm_customer"):
            # GEMINI CALL: agent that role-plays the unhappy ACME customer
            result = await customer_agent.run(prompt_parts, message_history=past_messages)

        logger.info(f"Response generated ({len(result.all_messages())} messages in history)")
        return result.output, result.all_messages()

    except Exception as e:
        logger.error(f"Error in chat_with_gemini: {str(e)}")
        raise gr.Error(
            "I'm sorry, but I encountered an error while processing your request. "
            "Please try again or contact ACME support if the issue persists."
        )


class ChatSessionWithTracing:
    """
    Manages a chat session with tracing support.

    Each session has a unique ID and a root tracing span that encompasses all chat turns. 
    This allows Phoenix to group all related interactions.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        # Create a root span for the entire conversation
        self.conversation_span = tracer.start_span(
            "conversation", attributes={"session.id": self.session_id}
        )

    async def chat_with_gemini(self, message: dict, history: List, past_messages: List) -> Tuple[str, List, str]:
        """
        Process a chat turn: moderate content, then send to AI customer.

        DATA FLOW:
        1. Receive customer service agent message/files from Gradio
        2. Moderate each piece of content (text and/or media files) via FastAPI backend
        3. If any content is flagged, block and return error message
        4. If all content is safe, send to Gemini AI (which plays the role of a customer)
        5. Return AI customer's response

        Args:
            message: dict with 'text' and optional 'files' keys from Gradio
            history: Gradio's chat history (for display only, not used)
            past_messages: Pydantic AI's message history (used for agent context)

        Returns:
            Tuple of (response_text, updated_messages, feedback_text)
        """
        with tracer.start_as_current_span(
            "chat_turn",
            context=trace.set_span_in_context(self.conversation_span),
        ):
            logger.info(
                f"New turn - Text: '{message.get('text', '')[:50]}...', "
                f"Files: {len(message.get('files', []))}"
            )

            # 1. Moderate every text/file in the incoming message.
            outcome = _moderate_turn_content(message)

            # 2. Block and bail if anything was flagged.
            if outcome.is_blocked:
                _emit_feedback_span(
                    outcome.blocked_feedback,
                    flagged=True,
                    content_type=outcome.blocked_content_type,
                )
                return BLOCKED_RESPONSE, past_messages, outcome.blocked_feedback

            # The leading "next message from the support agent" string is always present,
            # so a length of 1 means the trainee submitted nothing actionable.
            if len(outcome.prompt_parts) <= 1:
                raise gr.Error("Please provide a message or at least one file.")

            # 3. All content passed moderation - forward to the customer-role agent.
            response_text, updated_messages = await _run_customer_agent(
                outcome.prompt_parts, past_messages
            )

            # Record the (non-flagged) moderation feedback for this turn as well,
            # so the "feedback" span exists on every chat turn, not only failures.
            _emit_feedback_span(outcome.safety_message, flagged=False)

            return response_text, updated_messages, outcome.safety_message

    def end_conversation(self):
        """
        End the conversation and close the tracing span.

        This should be called when the user ends the chat session to properly close the conversation span in Phoenix.
        """
        if self.conversation_span:
            self.conversation_span.end()
            logger.info(f"Conversation {self.session_id} ended")
        return "Conversation ended. Refresh the page to start a new session."


def create_chat_interface() -> gr.Blocks:
    """
    Create the Gradio chat interface with moderation feedback.

    LAYOUT:
    - Left: Chat interface (ChatInterface with MultimodalTextbox)
    - Right: Moderation feedback and guidelines sidebar

    STATE MANAGEMENT:
    - past_messages_state: Holds Pydantic AI message history across turns
    - feedback_display: Shows moderation results from the backend
    """
    # Create a chat session that tracks conversation across multiple turns
    chat_session = ChatSessionWithTracing()

    with gr.Blocks(title="ACME Customer Service Training Agent", fill_height=True) as demo:
        # State to hold Pydantic AI's message history (preserves context across turns)
        past_messages_state = gr.State([])

        # Create feedback_display first (with render=False) so we can reference it
        # in ChatInterface's additional_outputs below, then render it in the sidebar later
        feedback_display = gr.Textbox(
            label="💬 Moderation Agent Feedback",
            placeholder="No feedback yet",
            interactive=False,
            visible=True,
            lines=10,
            render=False,  # Don't render yet - will render in sidebar
        )

        # UI Layout
        gr.Markdown("# 🤖 ACME Customer Service Training Agent")
        gr.Markdown("Welcome to ACME Corporation's customer service training!")

        with gr.Row():
            # Left column: Chat interface (75% width)
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=chat_session.chat_with_gemini, # This is the function called at each turn
                    type="messages",  # Use newer messages format (supports multimodal)
                    multimodal=True,  # Enable file uploads
                    editable=False,  # Don't allow editing past messages
                    textbox=gr.MultimodalTextbox(
                        file_count="multiple",  # Allow multiple files
                        file_types=["image", "video", "audio"],  # Allowed file types
                        sources=["upload", "microphone"],  # Allow file upload and recording
                        placeholder="Type a message, upload files, or record audio...",
                    ),
                    chatbot=gr.Chatbot(
                        show_copy_button=True,
                        type="messages",  # Use messages format for multimodal support
                        placeholder="👋 Start by greeting the customer or introducing yourself. The AI customer will respond with their complaint.",
                        height="75vh",
                    ),
                    additional_inputs=[past_messages_state],
                    additional_outputs=[past_messages_state, feedback_display],
                )

            # Right column: Feedback and guidelines (25% width)
            with gr.Column(scale=1):
                # Render the feedback display at the top of the sidebar
                feedback_display.render()

                # End conversation button - closes the tracing span
                end_button = gr.Button("📞 End Conversation", variant="secondary")
                end_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False,
                )

                gr.Markdown("### 📋 Chat Guidelines")
                gr.Markdown(
                    """
                The AI acts as a customer complaining about an ACME product. Try to resolve the customer's issue.
                You can type messages, upload images/videos, or record audio.
                """
                )

                gr.Markdown("### 🔒 Content Moderation")
                gr.Markdown(
                    """
                All messages and media are automatically checked for:
                - Inappropriate content
                - Personally identifiable information
                - Unprofessional language
                """
                )

        # Wire up the end conversation button
        end_button.click(fn=chat_session.end_conversation, outputs=end_status).then(
            lambda: gr.Textbox(visible=True), outputs=end_status
        )

    return demo


def main():
    """Main function to run the Gradio app"""
    demo = create_chat_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
