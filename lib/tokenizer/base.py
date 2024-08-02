from abc import ABC, abstractmethod
from typing import List, Dict


class BaseTokenizer(ABC):
    def __init__(self):
        """
        Base tokenizer to define the interface for all tokenizers.

        Attributes:
            token_to_item (Dict[int, str]): A mapping of token index to token.
            item_to_token (Dict[str, int]): A mapping of token to token index.
            current_index (int): The current index to keep track of token additions.
        """
        self.token_to_item: Dict[int, str] = {0: ""}
        self.item_to_token: Dict[str, int] = {"": 0}
        self.current_index: int = 1

    @abstractmethod
    def train(self, document: str) -> None:
        """
        Abstract method for training the tokenizer.

        Args:
            document (str): The training document used to build the vocabulary.
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Abstract method for encoding a text into tokens.

        Args:
            text (str): The input text to be encoded.

        Returns:
            List[int]: The list of encoded tokens as integers.
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Abstract method for decoding tokens back into text.

        Args:
            tokens (List[int]): The list of tokens to be decoded.

        Returns:
            str: The decoded text.
        """
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        """
        Abstract method to get the vocabulary size of the tokenizer.

        Returns:
            int: The vocabulary size.
        """
        pass
