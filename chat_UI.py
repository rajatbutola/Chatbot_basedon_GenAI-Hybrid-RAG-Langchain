#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import LLM_RAG
#####################################
# Streamlit UI
# Streamlit UI
def main():
    st.set_page_config(page_title="Chatbot", layout="wide")

    # Custom CSS for left/right alignment
    st.markdown("""
        <style>

        .chat-container {
            margin-bottom: 70px; /* Space for the fixed input area */
            padding: 10px;
            max-height: 75vh;
            overflow-y: auto;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .user-msg {
            text-align: right;
            margin: 10px 0;
            padding: 10px;
            background-color: #cef5d4;
            border-radius: 10px;
            display: inline-block;
            max-width: 100%;
            margin-left: auto;
        }
        .bot-msg {
            text-align: left;
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 10px;
            display: inline-block;
            max-width: 100%;
            margin-right: auto;
        }
        .input-container {
            position: fixed;
            bottom: 1rem;
            width: 100%;
            background-color: white;
            padding: 1rem;
        }
           
     /* Fixed input box at the bottom */
        .fixed-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 15px;
            border-top: 1px solid #ddd;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
            z-index: 1000;
    }
        /* Change the color of the Send button */
       div.stButton > button {
           background-color: #4CAF50; /* Green background */
           color: white; /* White text */
           border-radius: 8px;
           padding: 0.5rem 1rem;
           font-size: 16px;
           border: none;
           cursor: pointer;
       }
       div.stButton > button:hover {
           background-color: #45a049; /* Darker green on hover */
           color: white; /* White text */
       }
        </style>
        """, unsafe_allow_html=True)
    st.title("Chatbot 🤖")
    st.write("Ask me questions about ChimeraX commands!")
    # Initialize chat history and memory in session state
    if "memory" not in st.session_state:
        st.session_state.memory = LLM_RAG.memory
    if "dummy_key" not in st.session_state:
        st.session_state.dummy_key = 0  # Dummy key for resetting input field

    # Display Chat History
    st.markdown('<div>', unsafe_allow_html=True)
    for message in st.session_state.memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            st.markdown(f'<div class="user-msg">{message.content}</div>', unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.markdown(f'<div class="bot-msg">{message.content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input Container
    # Fixed input area
    

    ######################################
    with st.container():
        st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
        user_input = st.text_input(
            "Enter your query:",
            key=f"input_text_{st.session_state.dummy_key}",  # Unique key
            label_visibility="collapsed",
        )
        send_button = st.button("Send", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
###################################
    # Handle Send Button
    if send_button and user_input.strip():
        # Get response from RAG chain
        response = LLM_RAG.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]
                })

        
        #get_rag_response(user_input, st.session_state.memory.load_memory_variables({})["chat_history"])

        # Save to memory
        st.session_state.memory.save_context(
            {"input": user_input}, {"output": response["answer"]}
        )

        # Increment dummy_key to reset input field
        st.session_state.dummy_key += 1
        st.rerun()

    # Reset Chat History Button
    with st.sidebar:
        if st.button("Reset Chat"):
            st.session_state.memory.clear()
            st.session_state.dummy_key += 1  # Reset input field
            st.rerun()

if __name__ == "__main__":
    main()
