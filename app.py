import streamlit as st
from predictor import OPTKeyboardPredictor, VARIANT_METADATA

st.set_page_config(
    page_title="OPT Smart Mobile Keyboard",
    page_icon="📱",
    layout="centered",
)

# Custom styling for mobile keyboard suggestion bar
st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.55rem 0.75rem;
    }
    .keyboard-hint {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading OPT-125M keyboard model...")
def load_predictor(variant: str) -> OPTKeyboardPredictor:
    return OPTKeyboardPredictor.from_variant(variant=variant, device="cpu")


# Initialize session state
if "typed_text" not in st.session_state:
    st.session_state.typed_text = "My phone battery is at 1%, do you have a "
if "sent_messages" not in st.session_state:
    st.session_state.sent_messages = []


def apply_suggestion(word: str, mode: str):
    """Inserts next word or completes partial word and updates user cache."""
    current = st.session_state.typed_text
    if mode == "Starter" or not current:
        st.session_state.typed_text = f"{word} "
    elif mode == "Next-Word":
        st.session_state.typed_text = f"{current}{word} "
    else:
        # Replace partial prefix with completed word
        if " " in current:
            prefix_ctx, _ = current.rsplit(" ", 1)
            st.session_state.typed_text = f"{prefix_ctx} {word} "
        else:
            st.session_state.typed_text = f"{word} "


def set_sample_prompt(prompt_text: str):
    st.session_state.typed_text = prompt_text


def clear_input():
    st.session_state.typed_text = ""


# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Keyboard Engine Settings")
    variant_options = list(VARIANT_METADATA.keys())
    selected_variant = st.selectbox(
        "Model Variant",
        options=variant_options,
        format_func=lambda k: VARIANT_METADATA[k]["label"],
        index=0,
    )

    cache_weight = st.slider(
        "Personalization Boost (λ)",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Boosts words you frequently confirm or send in messages.",
    )

    st.divider()
    st.subheader("🧪 Quick Test Prompts")
    samples = [
        "My phone battery is at 1%, do you have a ",
        "Sure, I will meet you at the train ",
        "I am heading to the groc",
        "Hey! Are you free for lun",
    ]
    for idx, sample in enumerate(samples):
        st.button(
            f'Try: "{sample}"',
            key=f"sample_{idx}",
            on_click=set_sample_prompt,
            args=(sample,),
        )

# Load selected model variant
predictor = load_predictor(selected_variant)

# --- Main UI ---
st.title("📱 Context-Aware Mobile Keyboard")
st.caption(
    "BMDS2123 Natural Language Processing • `facebook/opt-125m` + LoRA + Hybrid Prefix-Aware Decoding"
)

# Conversation history display
if st.session_state.sent_messages:
    with st.container(border=True):
        st.markdown("**Sent Message History**")
        for msg in st.session_state.sent_messages[-5:]:
            st.markdown(f"💬 {msg}")

# Message composition box
st.text_area(
    "Type a message (end with a space for Next-Word prediction, or type partial letters for Prefix Completion):",
    key="typed_text",
    height=100,
    placeholder="Start typing here...",
)

# Generate Top-3 predictions
suggestions, latency_ms, mode = predictor.suggest(
    st.session_state.typed_text,
    top_k=3,
    cache_weight=cache_weight,
)

st.markdown(
    f"<div class='keyboard-hint'>Mode: <b>{mode}</b> • Inference Latency: <b>{latency_ms:.1f} ms</b> • Footprint: <b>{predictor.variant_info.get('size_label', '')}</b></div>",
    unsafe_allow_html=True,
)

# Top-3 Mobile Suggestion Bar
cols = st.columns(3)
for i in range(3):
    with cols[i]:
        if i < len(suggestions):
            word, prob = suggestions[i]
            st.button(
                f"{word}  ({prob * 100:.0f}%)",
                key=f"sugg_{i}_{word}",
                on_click=lambda w=word, m=mode: (
                    apply_suggestion(w, m),
                    predictor.update_user_cache(w),
                ),
            )
        else:
            st.button("—", key=f"empty_{i}", disabled=True)

# Action buttons (Send / Clear)
action_col1, action_col2 = st.columns([3, 1])
with action_col1:
    if st.button("Send Message 📤", type="primary"):
        msg = st.session_state.typed_text.strip()
        if msg:
            st.session_state.sent_messages.append(msg)
            predictor.update_user_cache(msg)
            st.session_state.typed_text = ""
            st.rerun()
with action_col2:
    st.button("Clear 🗑️", on_click=clear_input)

# Adaptive User Cache Viewer
with st.expander("🧠 Adaptive User Vocabulary Cache"):
    if predictor.user_freq_cache:
        sorted_cache = sorted(
            predictor.user_freq_cache.items(), key=lambda x: x[1], reverse=True
        )
        st.write(dict(sorted_cache))
        if st.button("Reset Personal Cache"):
            predictor.clear_user_cache()
            st.rerun()
    else:
        st.info("Tap suggested words or send messages to build your personalized word frequency cache.")
