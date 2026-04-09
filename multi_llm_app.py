"""
Multi-LLM Custom ChatGPT — Side-by-Side Answer Panel
IIT Patna AIML Certification — Project 1

Run with:  streamlit run multi_llm_app.py
Opens at:  http://localhost:8501

Note: Uses simulated LLM responses with distinct personas for each model.
      In production, replace simulate_response() with real API calls.
"""

import streamlit as st
import time
import random
import re
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-LLM ChatGPT — IIT Patna",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #f0f4f8; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stButton > button {
    background: white; color: #1a1a2e !important;
    font-weight: bold; border-radius: 8px; width: 100%;
    margin-bottom: 4px;
}

.model-header-openai {
    background: linear-gradient(135deg, #10a37f, #1a7a5e);
    color: white; padding: 10px 14px; border-radius: 10px 10px 0 0;
    font-weight: bold; font-size: 15px;
}
.model-header-claude {
    background: linear-gradient(135deg, #c96442, #a0522d);
    color: white; padding: 10px 14px; border-radius: 10px 10px 0 0;
    font-weight: bold; font-size: 15px;
}
.model-header-gemini {
    background: linear-gradient(135deg, #4285f4, #1a73e8);
    color: white; padding: 10px 14px; border-radius: 10px 10px 0 0;
    font-weight: bold; font-size: 15px;
}
.model-body {
    background: white; padding: 14px;
    border-radius: 0 0 10px 10px;
    border: 1px solid #dde8f0; border-top: none;
    min-height: 120px; font-size: 14px;
    line-height: 1.6; color: #333;
}
.user-bubble {
    background: #1a1a2e; color: white;
    padding: 10px 16px; border-radius: 18px 18px 4px 18px;
    margin: 8px 0; max-width: 75%; margin-left: auto;
    font-size: 14px;
}
.chat-bubble {
    padding: 10px 14px; border-radius: 4px 18px 18px 18px;
    margin: 4px 0; font-size: 14px; line-height: 1.6;
    border-left: 3px solid;
}
.chat-openai  { background:#f0faf7; border-color:#10a37f; color:#222; }
.chat-claude  { background:#fdf5f2; border-color:#c96442; color:#222; }
.chat-gemini  { background:#f0f4ff; border-color:#4285f4; color:#222; }

.selected-banner {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    color: white; padding: 8px 16px; border-radius: 8px;
    font-size: 13px; margin-bottom: 8px; text-align: center;
}
.header-banner {
    background: linear-gradient(90deg, #1a1a2e, #4285f4);
    color: white; padding: 18px 24px; border-radius: 12px;
    margin-bottom: 16px;
}
.phase-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 12px; font-size: 12px; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  LLM SIMULATION ENGINE
#  Each model has a distinct persona, vocabulary, and style.
#  In production: replace simulate_response() with real API calls.
# ══════════════════════════════════════════════════════════════════

PERSONAS = {
    "openai": {
        "name": "GPT-4o",
        "icon": "🟢",
        "color": "#10a37f",
        "style": "precise, concise, structured with bullet points when helpful",
        "intro": ["Here's a clear breakdown:", "Let me address this directly:",
                  "Great question.", "To answer this precisely:"],
        "tone": "professional and direct",
    },
    "claude": {
        "name": "Claude 3.5",
        "icon": "🟠",
        "color": "#c96442",
        "style": "thoughtful, nuanced, explores multiple angles",
        "intro": ["This is an interesting question to think through.",
                  "Let me explore this carefully.",
                  "There are a few dimensions worth considering here.",
                  "I'd approach this by first understanding the core idea:"],
        "tone": "reflective and thorough",
    },
    "gemini": {
        "name": "Gemini Pro",
        "icon": "🔵",
        "color": "#4285f4",
        "style": "creative, example-driven, uses analogies",
        "intro": ["Think of it this way:", "Great topic! Here's my take:",
                  "Let me paint a picture for you:",
                  "Imagine this scenario to understand:"],
        "tone": "engaging and example-rich",
    },
}

# Knowledge base for simulated answers
KNOWLEDGE = {
    "machine learning": {
        "openai": "Machine learning (ML) is a subset of AI that enables systems to learn from data.\n\n**Key types:**\n• Supervised learning — labeled data, predicts outputs\n• Unsupervised learning — finds patterns in unlabeled data\n• Reinforcement learning — learns via rewards and penalties\n\n**Common algorithms:** Linear regression, decision trees, neural networks, SVMs.",
        "claude": "Machine learning is fundamentally about teaching computers to improve through experience rather than explicit programming. What's fascinating is how it mirrors human learning — we don't hardcode rules, we expose the system to examples and let it discover patterns.\n\nThe three main paradigms each have their strengths: supervised learning excels when we have labeled examples, unsupervised learning reveals hidden structure in raw data, and reinforcement learning is ideal for sequential decision-making problems like games or robotics.",
        "gemini": "Imagine teaching a child to recognise cats — you don't give them a rulebook, you just show them thousands of cat photos! That's machine learning in a nutshell.\n\nML systems find patterns in data the same way. Show a model millions of emails labelled 'spam' or 'not spam', and it learns to classify new emails automatically. The three flavours are: supervised (learning from examples), unsupervised (finding hidden groups), and reinforcement (learning by trial and error — think training a dog with treats!).",
    },
    "python": {
        "openai": "Python is a high-level, interpreted programming language known for simplicity and readability.\n\n**Why Python?**\n• Clean, English-like syntax\n• Massive ecosystem (NumPy, Pandas, TensorFlow)\n• Rapid prototyping\n• Strong community support\n\n**Use cases:** Data science, web dev, automation, AI/ML.",
        "claude": "Python's rise to become the dominant language in AI and data science isn't accidental — it reflects a deliberate design philosophy that prioritises human readability over machine efficiency. Guido van Rossum designed Python to be code that reads almost like English prose.\n\nWhat makes Python particularly powerful in 2024 is its ecosystem: NumPy for numerical computing, Pandas for data manipulation, and PyTorch or TensorFlow for deep learning — all interoperable and actively maintained.",
        "gemini": "Python is like the Swiss Army knife of programming languages! It's the lingua franca of data science, AI, and web development.\n\nHere's a fun analogy: if programming languages were tools, C++ would be a precision scalpel (powerful but complex), Java would be a sturdy hammer (reliable, wordy), and Python would be a magic wand — a few lines of code and things just work. That's why 8 out of 10 AI researchers choose Python as their primary language!",
    },
    "rag": {
        "openai": "RAG (Retrieval-Augmented Generation) combines a retrieval system with a language model.\n\n**Pipeline:**\n1. Ingest documents → chunk text\n2. Embed chunks as vectors\n3. Store in vector database\n4. Query → retrieve top-k chunks\n5. Generate answer using retrieved context\n\n**Benefit:** Grounds LLM responses in factual documents, reducing hallucinations.",
        "claude": "RAG is one of the most elegant solutions to a fundamental LLM limitation — their knowledge is frozen at training time. By pairing a retrieval system with a generator, we get the best of both worlds: the fluency of a language model and the freshness of a live document store.\n\nWhat I find particularly interesting is how RAG changes the nature of LLM errors. Instead of confident hallucinations, a well-implemented RAG system either retrieves the right answer or admits it couldn't find relevant context — a much more trustworthy failure mode.",
        "gemini": "Imagine a brilliant student taking an open-book exam — that's RAG! Instead of relying purely on memorised knowledge, the student (LLM) can look up relevant pages (retrieved chunks) before writing the answer.\n\nThe magic is in the pipeline: your documents get chopped into bite-sized pieces, converted into mathematical 'meaning vectors', and stored in a searchable database. When you ask a question, the system finds the most relevant pieces (like finding the right page in a textbook) and hands them to the AI to write a precise, cited answer.",
    },
    "neural network": {
        "openai": "Neural networks are computational models inspired by the human brain.\n\n**Architecture:**\n• Input layer → Hidden layers → Output layer\n• Neurons connected by weighted edges\n• Activation functions introduce non-linearity\n\n**Training:** Backpropagation + gradient descent adjusts weights to minimise loss.\n\n**Types:** CNNs (images), RNNs (sequences), Transformers (language).",
        "claude": "Neural networks are perhaps the most profound example of bio-inspired computing. The core insight — that interconnected nodes processing weighted signals can learn complex functions — has proven remarkably general, powering everything from image recognition to language translation.\n\nWhat's often underappreciated is how much the training process resembles evolution: random initialisation, iterative refinement through gradient descent, and emergent capabilities that weren't explicitly programmed. The network discovers its own internal representations.",
        "gemini": "Think of a neural network as a team of detectives, each specialising in spotting different clues. The first layer notices basic shapes — edges and colours. The next layer combines those into textures. The deeper layers recognise complex patterns like 'fur' or 'wheels' — until the final layer confidently declares 'that's a cat!'.\n\nEach detective (neuron) learns from its mistakes via backpropagation — like a coach reviewing game footage and adjusting each player's strategy after every match.",
    },
    "api": {
        "openai": "An API (Application Programming Interface) is a contract that defines how software components communicate.\n\n**Key concepts:**\n• Endpoints — specific URLs that accept requests\n• HTTP methods — GET, POST, PUT, DELETE\n• Request/Response — structured data exchange (usually JSON)\n• Authentication — API keys, OAuth tokens\n\n**Example:** `POST /v1/messages` sends a prompt to Claude and returns a response.",
        "claude": "APIs are the invisible infrastructure of the modern web — every time you log in with Google, share to Twitter, or pay via Stripe, you're using APIs without realising it. They're essentially formal agreements between software systems about how to communicate.\n\nWhat makes a good API is thoughtful design: clear naming, consistent patterns, helpful error messages, and good documentation. The best APIs feel intuitive — you can almost guess how they work before reading the docs.",
        "gemini": "An API is like a restaurant menu! You (the developer) don't need to know how the kitchen works — you just order from the menu (API endpoints), specify your preferences (parameters), and the kitchen (server) sends back your dish (response).\n\nFor example, when your weather app shows today's forecast, it's 'ordering' from a weather API: 'Give me the forecast for Hyderabad please' → the weather server responds with temperature, humidity, and conditions. No kitchen secrets revealed!",
    },
}

def simulate_response(model_key: str, question: str, history: list) -> str:
    """
    Simulate a model response with distinct persona per model.
    Uses knowledge base for known topics, generates contextual
    responses for unknown queries.
    """
    q_lower = question.lower()
    persona = PERSONAS[model_key]

    # Check knowledge base
    for topic, answers in KNOWLEDGE.items():
        if topic in q_lower:
            return answers[model_key]

    # Context-aware response generation
    intro = random.choice(persona["intro"])

    if any(w in q_lower for w in ["what is", "explain", "define", "describe"]):
        topic_word = q_lower.replace("what is", "").replace("explain", "").replace("define","").replace("?","").strip()
        if model_key == "openai":
            return f"{intro}\n\n**{topic_word.title()}** is a concept with several important dimensions:\n\n• It provides a structured framework for solving problems in its domain\n• It has broad applications across industries and research fields\n• Understanding it requires grasping both theoretical foundations and practical applications\n\nFor a deeper dive, I'd recommend exploring foundational textbooks and hands-on projects."
        elif model_key == "claude":
            return f"{intro}\n\nWhen we talk about {topic_word}, we're entering a space where theory and practice intersect in fascinating ways. The concept has evolved significantly over the past decade, shaped by both academic research and real-world application.\n\nAt its core, the idea challenges us to think differently about how we approach problems — rather than applying rigid rules, it encourages adaptive, context-sensitive thinking."
        else:
            return f"{intro}\n\nLet me use an analogy to make {topic_word} crystal clear!\n\nImagine you're learning to ride a bicycle — you don't read a manual, you just try, fall, adjust, and improve. {topic_word.title()} works on a similar principle of iterative learning and adaptation.\n\nThe real-world applications are everywhere once you start looking — from recommendation systems to medical diagnosis!"

    elif any(w in q_lower for w in ["how", "why", "when"]):
        if model_key == "openai":
            return f"{intro}\n\nThe answer involves several key steps:\n\n1. **Understand the context** — identify the problem space\n2. **Apply the right methodology** — choose tools that fit the task\n3. **Iterate and validate** — test, measure, and refine\n4. **Document and share** — make results reproducible\n\nThis systematic approach ensures reliable, reproducible outcomes."
        elif model_key == "claude":
            return f"{intro}\n\nThis question deserves careful unpacking. The 'how' and 'why' are often more important than the 'what' — understanding the reasoning behind a process helps us adapt it when conditions change.\n\nThe short answer is that it works through a combination of structured methodology and adaptive feedback. But the longer answer involves appreciating why each step exists and what failure modes it prevents."
        else:
            return f"{intro}\n\nGreat question — let me break it down with a relatable example!\n\nThink of it like baking a cake 🎂 — you need the right ingredients (data), the right recipe (algorithm), the right temperature (hyperparameters), and patience (training time). Miss any one of these and the cake — or your model — won't rise!\n\nThe step-by-step process is actually quite logical once you see it through this lens."

    elif any(w in q_lower for w in ["difference", "compare", "vs", "versus", "better"]):
        if model_key == "openai":
            return f"{intro}\n\n| Aspect | Option A | Option B |\n|--------|----------|----------|\n| Speed | Faster | Slower |\n| Accuracy | Moderate | Higher |\n| Complexity | Lower | Higher |\n| Use case | Prototyping | Production |\n\n**Recommendation:** Choose based on your specific constraints — speed vs accuracy tradeoff is the key consideration."
        elif model_key == "claude":
            return f"{intro}\n\nComparisons like this are rarely black-and-white. Both approaches have legitimate use cases, and the 'better' option almost always depends on context: your data size, computational budget, latency requirements, and interpretability needs.\n\nI'd encourage thinking less about which is objectively better and more about which is better *for your specific problem*. The best practitioners I've observed are fluent in multiple approaches and choose deliberately."
        else:
            return f"{intro}\n\nOoh, a classic comparison question! Let me settle this with a sports analogy ⚽\n\nIt's like comparing a sprinter to a marathon runner — both are elite athletes, but optimised for completely different races. Similarly, these two approaches each shine in different scenarios.\n\nThe bottom line: use the sprinter (faster, simpler option) for quick prototypes and the marathon runner (slower, more robust option) when you need to go the distance in production!"

    else:
        # Generic thoughtful response
        if model_key == "openai":
            return f"To address your question about '{question}':\n\nThis touches on fundamental principles that span multiple domains. The key insight is that structured thinking combined with empirical validation yields the most reliable results.\n\n**Key takeaways:**\n• Start with clear problem definition\n• Apply proven methodologies\n• Validate with real data\n• Iterate based on feedback"
        elif model_key == "claude":
            return f"Your question about '{question}' opens up a genuinely rich area of inquiry.\n\nWhat strikes me most is how this topic sits at the intersection of theory and practice — the academic understanding and the applied reality often diverge in interesting ways. The most productive approach is usually to hold both perspectives simultaneously.\n\nI'd be happy to explore any specific aspect in more depth — the more specific the question, the more precise and useful the answer can be."
        else:
            return f"Love this question about '{question}'! 🚀\n\nHere's my favourite way to think about it: every complex topic has a simple core idea, and once you find it, everything else clicks into place.\n\nThe core idea here is about patterns — finding them, learning from them, and applying them in new contexts. That's what makes this field so exciting — the same fundamental insights keep showing up in surprising places!\n\nWant me to dive deeper into any specific aspect?"


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════

defaults = {
    "phase": "compare",          # "compare" or "single"
    "selected_model": None,
    "compare_history": [],       # [{question, responses:{model:text}, timestamp}]
    "single_history": [],        # [{role, content, model}]
    "current_responses": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🤖 Multi-LLM Chat")
    st.markdown("**IIT Patna — AIML Project 1**")
    st.markdown("---")

    st.markdown("### 📊 Mode")
    if st.session_state.phase == "compare":
        st.success("🔀 **Compare Mode** — All 3 models answer")
    else:
        m = st.session_state.selected_model
        p = PERSONAS[m]
        st.info(f"{p['icon']} **Single Mode** — Chatting with {p['name']}")

    st.markdown("---")
    st.markdown("### 🧠 Models")
    for key, p in PERSONAS.items():
        st.markdown(f"{p['icon']} **{p['name']}** — {p['tone']}")

    st.markdown("---")
    st.markdown("### 💡 Try These Questions")
    suggestions = [
        "What is machine learning?",
        "Explain neural networks",
        "What is RAG?",
        "How do APIs work?",
        "What is Python used for?",
        "Difference between supervised and unsupervised learning?",
    ]
    for sug in suggestions:
        if st.button(sug, key=f"sug_{sug}"):
            st.session_state["_pending"] = sug

    st.markdown("---")
    if st.button("🔀 Back to Compare Mode"):
        st.session_state.phase = "compare"
        st.session_state.selected_model = None
        st.session_state.single_history = []
        st.rerun()

    if st.button("🗑️ Clear All"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-banner">
  <h2 style="margin:0;color:white">🤖 Multi-LLM Custom ChatGPT</h2>
  <p style="margin:4px 0 0;opacity:0.85">
  IIT Patna AIML Certification · Project 1 · Side-by-Side Answer Panel
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PHASE A — COMPARE MODE (side-by-side)
# ══════════════════════════════════════════════════════════════════

if st.session_state.phase == "compare":

    st.markdown("### 🔀 Compare Mode — Ask once, get 3 answers side-by-side")
    st.caption("After seeing all responses, choose your favourite model to continue chatting with it.")

    # Show history
    for entry in st.session_state.compare_history:
        st.markdown(f'<div class="user-bubble">🙋 {entry["question"]}</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        cols = {"openai": c1, "claude": c2, "gemini": c3}
        for model_key, col in cols.items():
            p = PERSONAS[model_key]
            with col:
                st.markdown(
                    f'<div class="model-header-{model_key}">'
                    f'{p["icon"]} {p["name"]}</div>'
                    f'<div class="model-body">{entry["responses"][model_key]}</div>',
                    unsafe_allow_html=True)
        st.markdown("---")

    # Show current responses + continue buttons
    if st.session_state.current_responses:
        c1, c2, c3 = st.columns(3)
        cols = {"openai": c1, "claude": c2, "gemini": c3}
        for model_key, col in cols.items():
            p = PERSONAS[model_key]
            resp = st.session_state.current_responses.get(model_key, "")
            with col:
                st.markdown(
                    f'<div class="model-header-{model_key}">'
                    f'{p["icon"]} {p["name"]}</div>'
                    f'<div class="model-body">{resp}</div>',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"✅ Continue with {p['name']}",
                             key=f"pick_{model_key}", use_container_width=True):
                    # Save to history
                    last_q = st.session_state.get("_last_question", "")
                    st.session_state.compare_history.append({
                        "question": last_q,
                        "responses": dict(st.session_state.current_responses),
                        "timestamp": datetime.now().strftime("%H:%M"),
                    })
                    # Switch to single mode
                    st.session_state.phase = "single"
                    st.session_state.selected_model = model_key
                    # Seed single chat with the compare Q&A
                    st.session_state.single_history = [
                        {"role": "user", "content": last_q, "model": model_key},
                        {"role": "assistant",
                         "content": st.session_state.current_responses[model_key],
                         "model": model_key},
                    ]
                    st.session_state.current_responses = {}
                    st.rerun()

    # Input
    pending = st.session_state.pop("_pending", "")
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        question = st.text_input("Ask all 3 models:",
            value=pending,
            placeholder="e.g. What is machine learning?",
            label_visibility="collapsed", key="compare_input")
    with col_btn:
        ask = st.button("Ask All 🚀", use_container_width=True)

    if ask and question.strip():
        q = question.strip()
        st.session_state["_last_question"] = q
        # Save previous current_responses to history if any
        if st.session_state.current_responses and "_last_question" in st.session_state:
            pass  # Already handled on "Continue" click

        with st.spinner("Querying all 3 models in parallel..."):
            responses = {}
            for model_key in PERSONAS:
                time.sleep(0.3)  # Simulate API latency
                responses[model_key] = simulate_response(
                    model_key, q, st.session_state.compare_history)
            st.session_state.current_responses = responses
        st.rerun()


# ══════════════════════════════════════════════════════════════════
#  PHASE B — SINGLE MODEL CHAT
# ══════════════════════════════════════════════════════════════════

else:
    model_key = st.session_state.selected_model
    p = PERSONAS[model_key]

    st.markdown(
        f'<div class="selected-banner">'
        f'{p["icon"]} Continuing conversation with <b>{p["name"]}</b> — '
        f'{p["tone"]} · Use sidebar to go back to compare mode'
        f'</div>',
        unsafe_allow_html=True)

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.single_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">🙋 {msg["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-bubble chat-{model_key}">'
                    f'{p["icon"]} <b>{p["name"]}:</b><br>{msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True)

    # Suggested follow-ups
    st.markdown("**💡 Suggested follow-ups:**")
    followups = {
        "openai": ["Can you give an example?", "What are the limitations?",
                   "How is this used in industry?"],
        "claude": ["What's the philosophical implication?", "Where does this break down?",
                   "How has this evolved over time?"],
        "gemini": ["Give me a real-world analogy!", "What's the most exciting use case?",
                   "How would a beginner start learning this?"],
    }
    fu_cols = st.columns(3)
    for i, fu in enumerate(followups[model_key]):
        if fu_cols[i].button(fu, key=f"fu_{i}"):
            st.session_state["_pending_single"] = fu

    # Input
    pending_single = st.session_state.pop("_pending_single", "")
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        follow_q = st.text_input(
            f"Continue with {p['name']}:",
            value=pending_single,
            placeholder="Ask a follow-up question...",
            label_visibility="collapsed", key="single_input")
    with col_btn:
        send = st.button(f"Send {p['icon']}", use_container_width=True)

    if send and follow_q.strip():
        q = follow_q.strip()
        st.session_state.single_history.append(
            {"role": "user", "content": q, "model": model_key})
        with st.spinner(f"{p['name']} is thinking..."):
            time.sleep(0.5)
            answer = simulate_response(
                model_key, q, st.session_state.single_history)
        st.session_state.single_history.append(
            {"role": "assistant", "content": answer, "model": model_key})
        st.rerun()
