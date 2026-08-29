import streamlit as st
import requests


# ==================================================
# Configuration
# ==================================================

BACKEND_URL = "http://backend:8000/ollama/chat"


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="AI Assistant",
    page_icon="AI",
    layout="centered",
)


# ==================================================
# Title
# ==================================================

st.title("AI Assistant")
st.write("RAG-based AI Assistant with Tool Calling")


# ==================================================
# Chat History
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# Display Previous Messages
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================================
# User Input
# ==================================================

question = st.chat_input(
    "Ask something..."
)


# ==================================================
# Process Question
# ==================================================

if question:

    # ----------------------------------------------
    # Display user message
    # ----------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # ----------------------------------------------
    # Send request to FastAPI
    # ----------------------------------------------

    try:

        response = requests.post(
            BACKEND_URL,
            json={
                "question": question
            },
            timeout=60,
        )


        # ------------------------------------------
        # Check HTTP response
        # ------------------------------------------

        response.raise_for_status()

        data = response.json()


        # ------------------------------------------
        # Extract answer
        # ------------------------------------------

        answer = data.get(
            "answer",
            "No answer returned by the backend.",
        )


        # ------------------------------------------
        # Display assistant response
        # ------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(answer)


        # ------------------------------------------
        # Save assistant response
        # ------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


    except requests.exceptions.Timeout:

        with st.chat_message("assistant"):

            st.error(
                "The backend took too long to respond."
            )


    except requests.exceptions.ConnectionError:

        with st.chat_message("assistant"):

            st.error(
                "Could not connect to the AI backend."
            )


    except requests.exceptions.RequestException as e:

        with st.chat_message("assistant"):

            st.error(
                f"Backend error: {e}"
            )