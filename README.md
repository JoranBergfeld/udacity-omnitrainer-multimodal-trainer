# OmniTrainer: Multimodal Customer Service Trainer

This project holds a customer service trainer. It'll use various agents which will evaluate the customer service agent's performance. To get started, we should have an `.env` configured correctly. There's an `env.example` in the root of the repository. You need to configure the gemini endpoint, as well as the api key. The details are defined in the `env.example`.

## Functional Flow

A human trainee plays an ACME customer service agent in a chat. Every message and uploaded file is moderated *before* it reaches the simulated customer. Flagged content is blocked and explained back to the trainee; safe content is forwarded to an LLM that role-plays an unhappy customer. Every step is traced into Phoenix for later inspection.

```mermaid
sequenceDiagram
    actor Trainee as Trainee (CS agent)
    participant UI as Gradio Chat UI
    participant API as FastAPI Moderation API
    participant Mod as Moderation Agents<br/>(text / image / video / audio)
    participant Cust as Customer Agent (Gemini)
    participant Phx as Phoenix (tracing)

    Trainee->>UI: Send message and/or upload file(s)
    UI->>API: POST /api/v1/moderate_* (Bearer USER_API_KEY)
    API->>Mod: moderate_text / image / video / audio
    Mod->>Cust: (Gemini call) classify content
    Cust-->>Mod: ModerationResult (flags + rationale)
    Mod-->>API: ModerationResult
    API-->>UI: JSON result

    alt Content flagged (unsafe)
        UI-->>Trainee: ⚠️ Blocked + rationale feedback
        UI->>Phx: span "moderate_*" + "feedback"
    else Content safe
        UI->>Cust: customer_agent.run(prompt + history)
        Cust-->>UI: Customer reply
        UI-->>Trainee: Display customer reply
        UI->>Phx: spans "conversation" / "chat_turn" / "llm_customer"
    end
```

## Technical Architecture

The system runs as **two processes** plus an embedded tracing collector, started together by `multimodal-moderation` (`app.py`):

- **Gradio app** (`gradio_app.py`) — the front end and conversation orchestrator. It calls the moderation API over HTTP and the customer agent directly in-process.
- **FastAPI app** (`fastapi_app.py`) — a Bearer-authenticated moderation service exposing one endpoint per modality, each delegating to a Pydantic AI agent.
- **Phoenix** (`app.py` → `tracing.py`) — OpenTelemetry trace collector + UI on port 6006.

All agents target Google Gemini via Pydantic AI and return a typed `ModerationResult` subclass. Configuration (API keys, model, URLs) is centralized in `env.py`.

```mermaid
graph TD
    subgraph Client["Browser :7860"]
        Trainee["Trainee"]
    end

    subgraph GradioProc["Gradio process — gradio_app.py"]
        UI["gr.ChatInterface<br/>ChatSessionWithTracing"]
        CustAgent["customer_agent<br/>(Pydantic AI + Gemini)"]
    end

    subgraph APIProc["FastAPI process :8000 — fastapi_app.py"]
        Auth["Bearer auth<br/>validate_api_key"]
        EP["/moderate_text/<br/>/moderate_image_file/<br/>/moderate_video_file/<br/>/moderate_audio_file/"]
        TextA["text_agent"]
        ImgA["image_agent"]
        VidA["video_agent"]
        AudA["audio_agent"]
    end

    subgraph Shared["Shared modules"]
        Env["env.py<br/>(.env config)"]
        Types["types/moderation_result.py<br/>ModerationResult subclasses"]
    end

    Gemini["Google Gemini API"]
    Phoenix["Phoenix :6006<br/>(OpenTelemetry traces)"]

    Trainee -->|HTTP| UI
    UI -->|"HTTP POST + Bearer"| Auth
    Auth --> EP
    EP --> TextA & ImgA & VidA & AudA
    TextA & ImgA & VidA & AudA -->|classify| Gemini
    TextA & ImgA & VidA & AudA -->|validated output| Types
    UI -->|"if safe: run(prompt, history)"| CustAgent
    CustAgent -->|role-play customer| Gemini

    Env -.config.-> UI
    Env -.config.-> EP
    Env -.config.-> CustAgent
    UI -.spans.-> Phoenix
    CustAgent -.instrumented.-> Phoenix

    classDef proc fill:#eef,stroke:#557;
    class GradioProc,APIProc proc;
```

### Request lifecycle in brief

1. `app.py` launches Phoenix, then spawns the FastAPI and Gradio processes.
2. Gradio's `check_content_safety()` POSTs each text/media item to the matching FastAPI endpoint with the `USER_API_KEY` bearer token.
3. The endpoint runs the corresponding moderation agent, which prompts Gemini and parses the reply into a typed `ModerationResult` (`contains_pii`, `is_unfriendly`, `is_unprofessional`, `rationale`, plus modality-specific flags).
4. If any unsafe flag is set, Gradio blocks the message and surfaces the rationale; the `"feedback"` is recorded on the trace span. Otherwise the content (text + `BinaryContent` files) is passed to `customer_agent.run(...)` with prior message history.
5. Spans (`conversation` with `session.id`, `chat_turn`, `moderate_*`, `llm_customer`) stream to Phoenix at <http://localhost:6006>.
