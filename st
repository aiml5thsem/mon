import streamlit as st
import requests

st.set_page_config(page_title="Fast-Agent UI", page_icon="🤖", layout="centered")

st.title("💬 Fast-Agent MCP UI")
user_input = st.text_input("Enter your message:", "")

if st.button("Send") and user_input.strip():
    with st.spinner("Talking to agent..."):
        try:
            response = requests.post(
                "http://localhost:8001/mcp",
                json={"input": user_input},
                timeout=60
            )
            data = response.json()
            st.subheader("Agent Response")
            st.write(data.get("reply", "No reply found."))

            tools = data.get("tools_used", [])
            if tools:
                st.subheader("🛠 Tools Used")
                for t in tools:
                    st.write(f"- {t}")

        except Exception as e:
            st.error(f"Error: {e}")
