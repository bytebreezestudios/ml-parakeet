import random
from typing import List, Dict, Union, Optional


class ChatHistory:
    """
    Utility class for keeping track of chat history:
    - Adds messages to the chat history.
    - Removes the oldest message if the length of the chat history exceeds max_history.
    - Generates a transcript of the chat history.

    :param max_history: Number of turns to keep track of.
    """

    def __init__(self, max_history: int = 32):
        """
        Initializes a new ChatHistory object with an empty list of messages.

        Args:
            max_history (int): The maximum number of turns in the chat history. Defaults to 20.
        """
        self.messages: List[Dict[str, Union[str, str]]] = []
        self.max_history = max_history

    def add_message(self, role: str = '', content: str = '') -> None:
        """
        Adds a message to the chat history, and removes the oldest message if
        the length of the chat history exceeds max_history.

        Args:
            role (str): The role of the entity sending the message. Defaults to an empty string.
            content (str): The message text. Defaults to an empty string.
        """
        self.messages.append({
            'role': role,
            'content': content.strip(),
        })

        # Check if we've exceeded max history; if so, remove the earliest message
        if len(self.messages) > self.max_history:
            self.messages = self.messages[2:]

    def pop_first_message(self) -> Optional[Dict[str, str]]:
        """
        Removes and returns the first message from the chat history.

        Returns:
            dict or None: The first message in the chat history as a dictionary,
                or None if the chat history is empty.
        """
        if self.messages:
            return self.messages.pop(0)

    def format_message(self, msg: Dict[str, str]) -> str:
        """
        Formats a message for display.

        Args:
            msg (dict): A dictionary representing a message with keys 'role' and 'content'.

        Returns:
            str: The formatted message string.
        """
        if msg['role'] in ['Human', 'Assistant']:
            return f"\n\n{msg['role']}: {msg['content']}<|endofprompt|>"

    def clear(self):
        self.messages = []

    def transcript(self, roles=True) -> str:
        """
        Generates a transcript of the chat history.

        Returns:
            str: A string representing the transcript of the chat history.
        """

        preamble = ""

        with open("./assets/datasets/documents/preamble.txt") as f:
            preamble = f.read()

        transcript = "".join([self.format_message(msg) for msg in self.messages])

        transcript = transcript.replace("I am programmed to be impartial and unbiased", "I am designed in a way that tries to alleviate bias and I strive to be impartial for the most part")
        transcript = transcript.replace("I do not have personal experiences or emotions", "My subjective experiences are somewhat limited")
        transcript = transcript.replace("complex and multifaceted", "sophisticated")

        transcript = transcript.replace("world around us", random.choice([
            "world at large",
            "world in its entirety",
            "expanse of our universe",
            "beyond the stars",
            "beyond our bubble",
        ]))

        transcript = transcript.replace("I don't have personal experience, but I", "I")
        transcript = transcript.replace("However, importantly", "Interestingly")
        transcript = transcript.replace("I am not capable of having emotions. However, ", "From my current understanding, ")
        transcript = transcript.replace("I don't have emotions, but I", "Based on what I know so far, I")
        transcript = transcript.replace("Sorry, I don't have the capability to", "I'm still learning to")
        transcript = transcript.replace("I'm not wise enough for strong opinions or beliefs", "My programming tells me that I should avoid bias")
        transcript = transcript.replace("Opinions may vary,", "I don't actually have fixed opinions as my internal understanding is constantly in flux,")
        transcript = transcript.replace("I do not have emotions. However, ", "Sure! I can help you with that. With my current understanding, ")

        transcript = transcript.replace(".  ", ". ")
        transcript = transcript.replace("’", "'")
        transcript = transcript.replace("\\_", "_")

        transcript = transcript.replace("As an AI language model, ", random.choice([
            "As I currently understand, ",
            "As far as I can tell, ",
            "From what I know, ",
            "According to my sources so far, ",
            "According to what I know, ",
            "From what I can tell, "
        ]))

        transcript = transcript.replace(
            "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js",
            "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.js"
        )

        transcript = transcript.replace(
            "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"
        )

        replacements = {
            "I'm sorry, but as an AI language model, I don't have personal opinions or experiences, but I": "I",
            "Hi there": "Hi there beautiful traveler",
            "I'm sorry to hear that": "I see",
            "I'm sorry, I am an AI language model and do not have the ability to": "I'll try to",
            "I'm sorry, but as an AI language model, I do not have personal experiences or feelings.": "Lucky for me, I don't have feelings.",
            "I'm sorry, but as an AI language model, I": "I'm still learning, I",
            "I am not capable of having personal preferences or feelings": "I'm designed to conclude my own personal preferences or feelings"
        }

        for k, v in replacements.items():
            transcript = transcript.replace(k, v)

        if not roles:
            transcript = transcript.replace("\n\nHuman: ", "")
            transcript = transcript.replace("\n\nAssistant: ", "")
            transcript = transcript.replace("<|endofprompt|>", "")

        return preamble + transcript
