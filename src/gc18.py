#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeneCounsel-AI: A Multilingual Pharmacogenetic Counseling Education Platform
=========================================================================

This module implements an AI-powered educational platform for training pharmacy students
in pharmacogenetic counseling across multiple languages and therapeutic areas.

Created as part of PharmTutorAI at Virginia Commonwealth University School of Pharmacy.

Authors:
    Dayanjan S. Wijesinghe, Ph.D. (Principal Investigator)
    Lama Basalelah
    Autumn Brenner
    Krista Donohoe
    Lauren M. Caldas
    Suad Alshammari
    Walaa Abu Rukbah
    Monther Alsultan
    Silas Contaifer
    Kunal Modi

Copyright:
    Copyright (c) 2025 Dayanjan S. Wijesinghe
    PharmTutorAI.com
    School of Pharmacy, Virginia Commonwealth University

License:
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Version:
    1.0.0

Contact:
    Dayanjan S. Wijesinghe, Ph.D.
    Department of Pharmacotherapy and Outcome Sciences
    School of Pharmacy, Virginia Commonwealth University
    PO Box 980533
    Richmond, VA 23298-5048
    Email: wijesinghes@vcu.edu

This module is part of the PharmTutorAI's GeneCounsel-AI project, as described in:
Basalelah et al. (2025). Gene Counsel AI: Bridging Domestic and Global Language
Barriers in Pharmacy Education Through Large Language Model Implementation for
Pharmacogenetic Counseling. PLOS Digital Health.
"""

from typing import Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from openai import OpenAI
import streamlit as st
import logging
import time


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class AppConfig:
    """Application configuration and constants"""
    PAGE_TITLE = "GeneCounsel-AI - Dayanjan S. Wijesinghe - Copyright 2024"
    PAGE_ICON = ":dna:"
    HEADER_TEXT = """
        <font size='4'><b><i>This is an experimental AI being developed by PharmTutorAI and VCU School of Pharmacy 
        as an aid to PharmD education. This is provided for testing as is and is not meant to be deployed in an 
        education or clinical setting.</i></b></font>
    """
    DEFAULT_MODEL = "gpt-4o-2024-08-06"
    MODEL_MAPPING = {
        "GPT-4o": "gpt-4o-2024-08-06",
        "GPT-4o-mini": "gpt-4o-mini-2024-07-18"
    }
    MODEL_HELP = "GPT-4o points to gpt-4o-2024-08-06, GPT-4o-mini points to gpt-4o-mini-2024-07-18"
    LANGUAGE_OPTIONS = {
        "English": "English",
        "Español": "Spanish",
        "العربية": "Arabic",
        "Tiếng Việt": "Vietnamese",
        "中文": "Mandarin",
        "اردو": "Urdu",
        "Magyar": "Hungarian",
        "हिंदी": "Hindi",
        "తెలుగు": "Telugu",
        "Tagalog": "Filipino",
        "한국어": "Korean",
        "Русский": "Russian",
        "Français": "French",
        "Português": "Portuguese",
        "日本語": "Japanese"
    }
    CUSTOM_CSS = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .rtl {direction: rtl; text-align: right;}
        textarea {min-height: 100px !important; resize: vertical !important;}
        </style>
    """
    # Path configurations
    SRC_PATH = Path(__file__).parent
    PROJECT_ROOT = SRC_PATH.parent
    STREAMLIT_PATH = PROJECT_ROOT / '.streamlit'
    SECRETS_PATH = STREAMLIT_PATH / 'secrets.toml'

    # Application-specific paths
    BASE_PATH = SRC_PATH
    EVALUATION_RUBRIC_PATH = BASE_PATH / 'evaluation_rubric'
    MENTORSHIP_GUIDANCE_PATH = BASE_PATH / 'mentorship_guidance'
    SYSTEMMESSAGES_PATH = BASE_PATH / 'systemmessages'
class FileManager:
    """Handles all file operations"""

    @staticmethod
    def read_file(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path.absolute()}")
                return None
            return file_path.read_text(encoding=encoding)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    @staticmethod
    def get_files_in_directory(directory: Path, pattern: str = "*.txt") -> List[str]:
        try:
            if not directory.exists():
                logger.warning(f"Directory not found: {directory.absolute()}")
                return []
            return sorted([f.stem for f in directory.glob(pattern)])
        except Exception as e:
            logger.error(f"Error reading directory {directory}: {str(e)}")
            return []


class StateManager:
    """Manages application state"""

    @staticmethod
    def initialize_session_state():
        default_states = {
            'messages': [],
            'current_exercise': None,
            'evaluation': None,
            'current_language': "English",
            'current_model': "GPT-4o",
            'system_message': None,
            'evaluation_message_count': None,
            'form_submitted': False
        }

        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def reset_conversation():
        st.session_state.messages = []
        st.session_state.evaluation = None
        st.session_state.evaluation_message_count = None
        st.session_state.form_submitted = False

    @staticmethod
    def check_state_changes(new_language: str, new_exercise: str):
        if (st.session_state.get('current_language') != new_language or
                st.session_state.get('current_exercise') != new_exercise or
                st.session_state.get('current_model') != st.session_state.get('model_selector')):
            st.session_state.current_language = new_language
            st.session_state.current_exercise = new_exercise
            st.session_state.current_model = st.session_state.get('model_selector')
            StateManager.reset_conversation()


class APIKeyManager:
    """Manages API key retrieval with fallback to user input"""

    @staticmethod
    def get_api_key() -> Optional[str]:
        """
        Retrieves API key in following order:
        1. Check secrets.toml
        2. Check session state
        3. Prompt user input if neither above is available
        """
        # First try to get from secrets.toml if not already checked
        if 'secrets_key_checked' not in st.session_state:
            try:
                import toml
                if AppConfig.SECRETS_PATH.exists():
                    with open(AppConfig.SECRETS_PATH, 'r') as f:
                        secrets = toml.load(f)
                        if secrets.get('openai', {}).get('api_key'):
                            st.session_state.openai_api_key = secrets['openai']['api_key']
                            st.session_state.using_secrets_key = True
                st.session_state.secrets_key_checked = True
            except Exception as e:
                logger.error(f"Error reading secrets file: {str(e)}")
                st.session_state.secrets_key_checked = True

        # If no API key in session state, prompt user
        if not st.session_state.get('openai_api_key'):
            with st.sidebar:
                # Place at top of sidebar
                api_key_container = st.container()
                # Create a space to push other elements down
                st.markdown("---")

                # Go back to the container to render the API key input
                with api_key_container:
                    api_key = st.text_input(
                        "Enter OpenAI API Key",
                        type="password",
                        help="No API key found in configuration. Please enter your OpenAI API key to continue.",
                        placeholder="sk-...",
                        key="api_key_input"
                    )

                    if api_key:
                        if api_key.startswith('sk-') and len(api_key) > 20:
                            st.session_state.openai_api_key = api_key
                            st.session_state.using_secrets_key = False
                            st.success("API key saved for this session!")
                        else:
                            st.error("Please enter a valid OpenAI API key")
                            return None
                    else:
                        st.warning("Please enter your OpenAI API key to continue")
                        st.stop()

        return st.session_state.get('openai_api_key')


class OpenAIService:
    """Handles all OpenAI API interactions"""

    def __init__(self):
        try:
            # Get API key with fallback strategy
            api_key = APIKeyManager.get_api_key()
            if not api_key:
                raise ValueError("No valid API key available")

            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI service: {str(e)}")
            raise

    def generate_chat_completion(
            self,
            messages: List[Dict[str, str]],
            model: str = AppConfig.DEFAULT_MODEL,
            temperature: float = 0.7
    ) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": msg["role"], "content": str(msg["content"])} for msg in messages],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if "Invalid API key" in str(e):
                # Clear the invalid API key from session state only if it's not from secrets
                if not st.session_state.get('using_secrets_key', False):
                    st.session_state.openai_api_key = None
                    st.error("Invalid API key. Please enter a valid OpenAI API key.")
                else:
                    st.error("Invalid API key in configuration file.")
                st.stop()
            logger.error(f"Error in OpenAI API call: {str(e)}")
            return None


class EvaluationManager:
    """Manages conversation evaluation"""

    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    def handle_evaluation(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            current_language = AppConfig.LANGUAGE_OPTIONS[st.session_state.current_language]

            # Load components
            mentorship_guidance = FileManager.read_file(AppConfig.MENTORSHIP_GUIDANCE_PATH / 'guidance.txt')
            if mentorship_guidance:
                mentorship_guidance = mentorship_guidance.replace("{language}", current_language)

            exercise_content = messages[0]['content']
            evaluation_rubric = FileManager.read_file(AppConfig.EVALUATION_RUBRIC_PATH / 'rubric.txt')
            if not evaluation_rubric:
                logger.error("Failed to load evaluation rubric")
                return None

            # Store evaluation point
            st.session_state.evaluation_message_count = len(messages)

            # Build system message with language context
            system_context = (
                f"{exercise_content}\n\n"
                f"Additional Context for Language and Evaluation:\n"
                f"- All evaluation feedback must be provided in {current_language}\n"
                f"- All criteria descriptions must be in {current_language}\n"
                f"- All feedback and explanations must be in {current_language}\n\n"
                f"Mentorship Guidance:\n{mentorship_guidance}\n\n"
                f"Evaluation Framework:\n{evaluation_rubric}"
            )

            # Create conversation history
            conversation_history = [
                {"role": "system", "content": system_context}
            ]
            conversation_history.extend(messages[1:])

            # Add evaluation request
            eval_request = (
                f"Please evaluate this conversation. "
                f"Provide all feedback, scoring, and explanations in {current_language}."
            )
            conversation_history.append({"role": "user", "content": eval_request})

            # Generate evaluation
            return self.openai_service.generate_chat_completion(
                messages=conversation_history,
                temperature=0.0
            )
        except Exception as e:
            logger.error(f"Error generating evaluation: {str(e)}")
            return None


class ChatManager:
    """Manages chat operations"""

    def __init__(self, openai_service: OpenAIService):
        self.openai_service = openai_service

    def handle_chat(self, user_input: str, system_message: str) -> bool:
        try:
            if not st.session_state.messages:
                st.session_state.messages = [{"role": "system", "content": system_message}]

            if not st.session_state.messages[-1].get("content") == user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})

            messages = self._prepare_messages(user_input)
            selected_model = st.session_state.get('model_selector', 'GPT-4o')
            model_name = AppConfig.MODEL_MAPPING.get(selected_model, AppConfig.DEFAULT_MODEL)

            assistant_response = self.openai_service.generate_chat_completion(messages, model_name)
            if assistant_response:
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                return True
            return False
        except Exception as e:
            logger.error(f"Error in chat handling: {str(e)}")
            return False

    def _prepare_messages(self, user_input: str) -> List[Dict[str, str]]:
        if st.session_state.evaluation:
            return self._prepare_evaluation_messages(user_input)
        messages = st.session_state.messages.copy()
        messages.append({"role": "user", "content": user_input})
        return messages

    def _prepare_evaluation_messages(self, user_input: str) -> List[Dict[str, str]]:
        mentorship_guidance = FileManager.read_file(AppConfig.BASE_PATH / 'mentorship_guidance' / 'guidance.txt')
        exercise_content = st.session_state.system_message
        rubric = FileManager.read_file(AppConfig.BASE_PATH / 'evaluation_rubric' / 'rubric.txt')

        enhanced_context = (
            f"{exercise_content}\n\n"
            f"Additional Context for Feedback:\n"
            f"{mentorship_guidance}\n\n"
            f"Evaluation Criteria:\n{rubric}\n\n"
            f"Current Evaluation:\n{st.session_state.evaluation}"
        )

        messages = [{"role": "system", "content": enhanced_context}]
        messages.extend(st.session_state.messages[1:])
        messages.append({"role": "user", "content": user_input})
        return messages

    def generate_conversation_text(self) -> bytes:
        """Generate text file from conversation history"""
        output = []

        # Add header
        output.append("PharmTutorAI Conversation Record")
        output.append("=" * 30)
        output.append("")

        # Add metadata
        output.append("Conversation Details")
        output.append("-" * 20)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        exercise_name = st.session_state.get('current_exercise', 'Unknown Exercise')
        selected_language = st.session_state.get('current_language', 'English')

        output.append(f"Date: {current_time}")
        output.append(f"Exercise: {exercise_name}")
        output.append(f"Language: {selected_language}")
        output.append("")

        # Add pre-evaluation conversation
        output.append("Initial Conversation")
        output.append("-" * 18)

        evaluation_count = st.session_state.get('evaluation_message_count')

        if evaluation_count:
            # Add pre-evaluation messages
            for msg in st.session_state.messages[1:evaluation_count]:
                if msg["role"] != "system":
                    role = "You: " if msg["role"] == "user" else "Assistant: "
                    message_lines = msg["content"].split('\n')
                    output.append(role + message_lines[0])
                    for line in message_lines[1:]:
                        output.append(" " * len(role) + line)
                    output.append("")

            # Add evaluation
            if st.session_state.evaluation:
                output.append("")
                output.append("Evaluation Results")
                output.append("-" * 18)
                eval_lines = st.session_state.evaluation.split('\n')
                output.extend(eval_lines)
                output.append("")

            # Add post-evaluation messages if any exist
            if len(st.session_state.messages) > evaluation_count:
                output.append("Post-Evaluation Conversation")
                output.append("-" * 26)
                for msg in st.session_state.messages[evaluation_count:]:
                    if msg["role"] != "system":
                        role = "You: " if msg["role"] == "user" else "Assistant: "
                        message_lines = msg["content"].split('\n')
                        output.append(role + message_lines[0])
                        for line in message_lines[1:]:
                            output.append(" " * len(role) + line)
                        output.append("")
        else:
            # No evaluation yet, just add all messages
            for msg in st.session_state.messages[1:]:
                if msg["role"] != "system":
                    role = "You: " if msg["role"] == "user" else "Assistant: "
                    message_lines = msg["content"].split('\n')
                    output.append(role + message_lines[0])
                    for line in message_lines[1:]:
                        output.append(" " * len(role) + line)
                    output.append("")

        return '\n'.join(output).encode('utf-8')


class MessageDisplayManager:
    """Manages the display of chat messages"""

    @staticmethod
    def display_messages():
        if not st.session_state.messages:
            return

        evaluation_count = st.session_state.get('evaluation_message_count')

        if evaluation_count is not None:
            # Post-evaluation messages
            post_eval_messages = list(reversed(st.session_state.messages[evaluation_count:]))
            for message in post_eval_messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.write(message["content"])

            # Display evaluation
            if st.session_state.evaluation:
                st.markdown("### Evaluation Result")
                st.markdown(
                    f'<div style="background-color: #9B59B6; color: white; padding: 10px; border-radius: 10px;">'
                    f'{st.session_state.evaluation}</div>',
                    unsafe_allow_html=True
                )

            # Pre-evaluation messages
            pre_eval_messages = list(reversed(st.session_state.messages[1:evaluation_count]))
            for message in pre_eval_messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
        else:
            # No evaluation yet
            for message in reversed(st.session_state.messages[1:]):
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.write(message["content"])


class UIComponents:
    """Manages UI components and rendering"""

    @staticmethod
    def setup_page():
        st.set_page_config(
            page_title=AppConfig.PAGE_TITLE,
            page_icon=AppConfig.PAGE_ICON
        )
        st.markdown(AppConfig.CUSTOM_CSS, unsafe_allow_html=True)
        st.markdown(AppConfig.HEADER_TEXT, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar() -> Tuple[str, str]:
        with st.sidebar:
            selected_exercise = UIComponents._render_exercise_selector()
            selected_language = UIComponents._render_language_selector()
            UIComponents._render_model_selector()
            UIComponents._render_conversation_controls()
            UIComponents._render_footer()
            return selected_language, selected_exercise

    @staticmethod
    def _render_exercise_selector() -> str:
        st.subheader("Select Exercise")
        available_messages = FileManager.get_files_in_directory(AppConfig.SYSTEMMESSAGES_PATH)
        if not available_messages:
            st.error("No exercise files found in systemmessages directory")
            return ""

        selected_exercise = st.selectbox(
            "Select the exercise you want to attempt:",
            options=available_messages,
            key='exercise_selector'
        )

        # Load and validate system message when exercise is selected
        if selected_exercise:
            system_message = FileManager.read_file(AppConfig.SYSTEMMESSAGES_PATH / f"{selected_exercise}.txt")
            if system_message:
                current_language = st.session_state.get('current_language', 'English')
                system_message = system_message.replace(
                    "{language}",
                    AppConfig.LANGUAGE_OPTIONS.get(current_language, "English")
                )
                if system_message != st.session_state.get('system_message'):
                    st.session_state.system_message = system_message
                    st.session_state.messages = [{"role": "system", "content": system_message}]
                st.success(f"Loaded exercise: {selected_exercise}")
            else:
                st.error("Failed to load selected exercise")
        return selected_exercise

    @staticmethod
    def _render_language_selector() -> str:
        st.markdown("---")
        st.subheader("Select Language")
        return st.radio(
            "Select a language",
            options=list(AppConfig.LANGUAGE_OPTIONS.keys()),
            key='language_selector',
            label_visibility="collapsed"
        )

    @staticmethod
    def _render_model_selector():
        st.markdown("---")
        st.subheader("Select Model")
        st.radio(
            "Choose a model:",
            options=list(AppConfig.MODEL_MAPPING.keys()),
            help=AppConfig.MODEL_HELP,
            key='model_selector'
        )

    @staticmethod
    def _render_conversation_controls():
        st.markdown("---")

        # Clear conversation button
        if st.button("Clear Conversation", key="clear_conversation_btn"):
            StateManager.reset_conversation()
            st.rerun()

        # Evaluate conversation button
        if st.button("Evaluate Conversation", key="evaluate_conversation_btn"):
            if len(st.session_state.messages) > 1:
                evaluation_manager = EvaluationManager(OpenAIService())
                evaluation_result = evaluation_manager.handle_evaluation(st.session_state.messages)
                if evaluation_result:
                    st.session_state.evaluation = evaluation_result
                    st.session_state.evaluation_message_count = len(st.session_state.messages)
                    st.rerun()
                else:
                    st.error("Failed to generate evaluation")
            else:
                st.warning("Please have a conversation first before evaluating")

        # Download conversation button
        if st.button("Download Conversation", key="download_conversation_btn"):
            if len(st.session_state.messages) > 1:
                chat_manager = ChatManager(OpenAIService())
                text_bytes = chat_manager.generate_conversation_text()
                exercise_name = st.session_state.get('current_exercise', 'unknown')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{exercise_name}_{timestamp}.txt"

                st.download_button(
                    label="Download Text",
                    data=text_bytes,
                    file_name=filename,
                    mime="text/plain",
                    key="download_text_btn"
                )
            else:
                st.warning("Please have a conversation first before downloading")



    @staticmethod
    def _render_footer():
        st.markdown("---")
        st.markdown(
            """<div style='text-align: center; font-size: small;'>
                &copy; 2025 Dayanjan S. Wijesinghe. All rights reserved.
            </div>""",
            unsafe_allow_html=True
        )


def main():
    """Main application entry point"""
    # Initialize components
    UIComponents.setup_page()
    StateManager.initialize_session_state()
    openai_service = OpenAIService()
    chat_manager = ChatManager(openai_service)

    # Render UI and get selected options
    selected_language, selected_exercise = UIComponents.render_sidebar()

    # Check for state changes
    StateManager.check_state_changes(selected_language, selected_exercise)

    # Regular chat form for manual input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "You:",
            key="user_input",
            height=100
        )
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            success = chat_manager.handle_chat(user_input, st.session_state.system_message)
            if success:
                st.rerun()
            else:
                st.error("Failed to generate response. Please try again.")

    # Display chat content
    MessageDisplayManager.display_messages()


if __name__ == "__main__":
    main()
