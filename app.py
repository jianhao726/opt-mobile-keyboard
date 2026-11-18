import streamlit as st
from predictor import OPTKeyboardPredictor, VARIANT_METADATA

st.set_page_config(
    page_title="OPT Smart Mobile Keyboard",
    page_icon="📱",
    layout="centered",
)

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


# --- Session State Initialization ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Single-User"
if "typed_text" not in st.session_state:
    st.session_state.typed_text = "My phone battery is at 1%, do you have a "
if "solo_messages" not in st.session_state:
    st.session_state.solo_messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"sender": "User A", "text": "Hey! Are you free for lunch today?"}
    ]
if "active_speaker" not in st.session_state:
    st.session_state.active_speaker = "User B"
if "user_caches" not in st.session_state:
    st.session_state.user_caches = {
        "Solo": {},
        "User A": {},
        "User B": {},
    }


def apply_suggestion(word: str, mode: str):
    """Inserts next word or completes partial word inside the active draft box."""
    current = st.session_state.typed_text
    if mode == "Starter" or not current.strip():
        st.session_state.typed_text = f"{word} "
    elif mode == "Next-Word":
        st.session_state.typed_text = f"{current}{word} "
    else:
        if " " in current:
            prefix_ctx, _ = current.rsplit(" ", 1)
            st.session_state.typed_text = f"{prefix_ctx} {word} "
        else:
            st.session_state.typed_text = f"{word} "


def set_sample_prompt(prompt_text: str):
    st.session_state.typed_text = prompt_text


def start_two_user_scenario(opening_msg: str, reply_prefix: str = ""):
    st.session_state.chat_history = [{"sender": "User A", "text": opening_msg}]
    st.session_state.active_speaker = "User B"
    st.session_state.typed_text = reply_prefix


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
        help="Boosts words frequently confirmed or sent by the active user.",
    )

    st.divider()
    if st.session_state.app_mode == "Single-User":
        st.subheader("🧪 Single-User Test Prompts")
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
    else:
        st.subheader("💬 2-User Dialogue Scenarios")
        scenarios = [
            ("Lunch Invite", "Hey! Are you free for lunch today?", ""),
            ("Phone Battery", "My phone battery is at 1%, do you have a charger?", "Yes, I have a "),
            ("Train Meetup", "I just arrived at the station, where are you?", "I am on my "),
        ]
        for idx, (label, msg_a, draft_b) in enumerate(scenarios):
            st.button(
                f"Scenario: {label}",
                key=f"scen_{idx}",
                on_click=start_two_user_scenario,
                args=(msg_a, draft_b),
            )

# Load cached model (no reload needed when switching users or modes)
predictor = load_predictor(selected_variant)

# --- Header & Mode Selector ---
st.title("📱 Context-Aware Mobile Keyboard")
st.caption(
    "BMDS2123 Natural Language Processing • `facebook/opt-125m` + LoRA + Hybrid Prefix-Aware Decoding"
)

current_label = "Single-User" if st.session_state.app_mode == "Single-User" else "Multi-User (2-Person Chat)"
mode_choice = st.segmented_control(
    "Select Interaction Mode",
    options=["Single-User", "Multi-User (2-Person Chat)"],
    default=current_label,
    selection_mode="single",
)
if mode_choice is None:
    mode_choice = current_label

new_mode = "Multi-User" if mode_choice == "Multi-User (2-Person Chat)" else "Single-User"
if new_mode != st.session_state.app_mode:
    st.session_state.app_mode = new_mode
    st.session_state.typed_text = "" if new_mode == "Multi-User" else "My phone battery is at 1%, do you have a "
    st.rerun()

st.divider()

# ==============================================================================
# MODE 1: SINGLE-USER (SOLO TYPING)
# ==============================================================================
if st.session_state.app_mode == "Single-User":
    active_profile = "Solo"
    predictor.user_freq_cache = st.session_state.user_caches[active_profile]

    if st.session_state.solo_messages:
        with st.container(border=True):
            st.markdown("**Sent Message History**")
            for msg in st.session_state.solo_messages[-5:]:
                st.markdown(f"💬 {msg}")

    st.text_area(
        "Type a message (end with a space for Next-Word prediction, or type partial letters for Prefix Completion):",
        key="typed_text",
        height=100,
        placeholder="Start typing here...",
    )

    model_input = st.session_state.typed_text
    send_label = "Send Message 📤"

# ==============================================================================
# MODE 2: MULTI-USER (2-PERSON CONVERSATION)
# ==============================================================================
else:
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        speaker = st.segmented_control(
            "Active Typist",
            options=["User A", "User B"],
            default=st.session_state.active_speaker,
            selection_mode="single",
        )
        if speaker:
            st.session_state.active_speaker = speaker
    with ctrl_col2:
        auto_switch = st.checkbox("Auto-switch turn", value=True)

    active_profile = st.session_state.active_speaker
    predictor.user_freq_cache = st.session_state.user_caches[active_profile]

    # Chat message thread
    with st.container(border=True):
        if not st.session_state.chat_history:
            st.caption("No messages yet. Start typing as User A below!")
        else:
            for item in st.session_state.chat_history[-8:]:
                is_user_a = item["sender"] == "User A"
                role = "user" if is_user_a else "assistant"
                avatar = "🧑" if is_user_a else "👩"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(f"**{item['sender']}:** {item['text']}")

    badge = "🧑 User A" if active_profile == "User A" else "👩 User B"
    st.text_area(
        f"Replying as {badge} (predictions adapt to conversation context):",
        key="typed_text",
        height=90,
        placeholder=f"Type {active_profile}'s message here...",
    )

    # Condition predictions on recent chat turns + active user's current draft
    draft = st.session_state.typed_text
    if st.session_state.chat_history:
        recent_turns = " ".join(m["text"].strip() for m in st.session_state.chat_history[-2:])
        if not draft.strip():
            model_input = f"{recent_turns} "
        else:
            model_input = f"{recent_turns} {draft}"
    else:
        model_input = draft

    send_label = f"Send as {active_profile} 📤"

# ==============================================================================
# SHARED TOP-3 MOBILE KEYBOARD BAR & ACTIONS
# ==============================================================================
suggestions, latency_ms, mode = predictor.suggest(
    model_input,
    top_k=3,
    cache_weight=cache_weight,
)

st.markdown(
    f"<div class='keyboard-hint'>Profile: <b>{active_profile}</b> • Mode: <b>{mode}</b> • Latency: <b>{latency_ms:.1f} ms</b> • Footprint: <b>{predictor.variant_info.get('size_label', '')}</b></div>",
    unsafe_allow_html=True,
)

cols = st.columns(3)
for i in range(3):
    with cols[i]:
        if i < len(suggestions):
            word, prob = suggestions[i]
            st.button(
                f"{word}  ({prob * 100:.0f}%)",
                key=f"sugg_{active_profile}_{i}_{word}",
                on_click=lambda w=word, m=mode: (
                    apply_suggestion(w, m),
                    predictor.update_user_cache(w),
                ),
            )
        else:
            st.button("—", key=f"empty_{i}", disabled=True)

action_col1, action_col2, action_col3 = st.columns([3, 1, 1])
with action_col1:
    if st.button(send_label, type="primary"):
        msg = st.session_state.typed_text.strip()
        if msg:
            predictor.update_user_cache(msg)
            if st.session_state.app_mode == "Single-User":
                st.session_state.solo_messages.append(msg)
            else:
                st.session_state.chat_history.append({"sender": active_profile, "text": msg})
                if auto_switch:
                    st.session_state.active_speaker = "User B" if active_profile == "User A" else "User A"
            st.session_state.typed_text = ""
            st.rerun()
with action_col2:
    st.button("Clear 🗑️", on_click=clear_input)
with action_col3:
    if st.session_state.app_mode == "Multi-User":
        if st.button("Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.typed_text = ""
            st.rerun()

# Adaptive User Cache Viewer (per active user profile)
with st.expander(f"🧠 Adaptive Vocabulary Cache ({active_profile})"):
    active_cache = st.session_state.user_caches[active_profile]
    if active_cache:
        sorted_cache = sorted(active_cache.items(), key=lambda x: x[1], reverse=True)
        st.write(dict(sorted_cache))
        if st.button(f"Reset {active_profile}'s Cache"):
            st.session_state.user_caches[active_profile].clear()
            st.rerun()
    else:
        st.info(f"Tap suggested words or send messages as {active_profile} to build their personal word frequency cache.")
