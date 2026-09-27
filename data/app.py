import streamlit as st
from chatbot import build_chain
from langchain_core.runnables import RunnableConfig

st.set_page_config(page_title="NovaBot", page_icon="💬")
st.title("💬 NovaBot — NovaCorp Assistant")

# Build the chain once and cache it across reruns (Streamlit reruns the
# whole script on every interaction, so this avoids reloading the model
# and vector store every single time).
@st.cache_resource
def get_chain():
    return build_chain()

chain = get_chain()

# Streamlit's own session state, separate from LangChain's memory store —
# used here just to redraw the chat bubbles on screen.
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Redraw prior messages
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask NovaBot something about NovaCorp...")

if question:
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config: RunnableConfig = {
                "configurable": {"session_id": st.session_state.session_id}
            }
            answer = chain.invoke({"question": question}, config=config)
        st.markdown(answer)

    st.session_state.display_messages.append({"role": "assistant", "content": answer})