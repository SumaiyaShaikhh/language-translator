import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

translator = Agent(
    name='Translator Agent',
    instructions="""
You are a smart and accurate language translator.

- Detect if the input is in English or Roman Urdu (Urdu written in Latin script).
- If the input is in English, translate it into Roman Urdu, preserving correct **pronoun roles and tense**.
- If the input is in Roman Urdu, translate it into proper English.
- Only return the translation — no extra text or explanation.
"""
)

st.set_page_config(page_title="English ↔ Roman Urdu Translator", layout="centered")

st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 12px;
    }
    .stButton > button {
        font-size: 18px;
        padding: 12px 24px;
    }
    @media (max-width: 600px) {
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .stButton > button {
            font-size: 16px;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Smart Language Translator")
st.subheader("Your Bridge Between English and Roman Urdu")

user_input = st.text_input("Enter text to translate", "")

if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "translate_result" not in st.session_state:
    st.session_state.translate_result = ""

button_clicked = st.button("Translate")
input_changed = user_input.strip() and user_input != st.session_state.last_input

if (button_clicked or input_changed) and user_input.strip():
    with st.spinner("Translating..."):
        async def translate():
            return await Runner.run(translator, input=user_input, run_config=config)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(translate())
            st.session_state.translate_result = response.final_output
            st.session_state.last_input = user_input
        finally:
            loop.close()

if st.session_state.translate_result:
    st.success("Translation:")
    st.write(f"**{st.session_state.translate_result}**")
