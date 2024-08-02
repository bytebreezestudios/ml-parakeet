import tiktoken
from tiktoken import get_encoding

from typing import List
from lib.tokenizer.base import BaseTokenizer


class TikTokenizer(BaseTokenizer):
    def __init__(self, encoding: str = "cl100k_base"):
        """
        TikTokenizer class, another tokenizer based on some encoding.

        Args:
            encoding (str, optional): The type of encoding to use. Defaults to "cl100k_base".
        """

        """
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        
        # In production, load the arguments directly instead of accessing private attributes
        # See openai_public.py for examples of arguments for specific encodings
        enc = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens,
                "<|im_start|>": 100264,
                "<|im_end|>": 100265,
            }
        )
        
        # GPT3.5 (cl100k_base) Specific.
        # {
        #     '<|endoftext|>': 100257,
        #     '<|fim_prefix|>': 100258,
        #     '<|fim_middle|>': 100259,
        #     '<|fim_suffix|>': 100260,
        #     '<|endofprompt|>': 100276
        # }
        """

        super().__init__()

        if encoding == "cl100k_base":
            enc_base = get_encoding(encoding)

            self.enc = tiktoken.Encoding(
                name=f"{encoding}_prk",
                pat_str=enc_base._pat_str,
                mergeable_ranks=enc_base._mergeable_ranks,
                special_tokens={
                    **enc_base._special_tokens,
                    "<|padding|>": enc_base.max_token_value + 1,
                    "\n\nSystem: ": enc_base.max_token_value + 2,
                    "\n\nHuman: ": enc_base.max_token_value + 3,
                    "\n\nAssistant: ": enc_base.max_token_value + 4,
                }
            )
        else:
            enc_base = tiktoken.get_encoding(encoding)

            self.enc = tiktoken.Encoding(
                name=f"{encoding}_prk",
                pat_str=enc_base._pat_str,
                mergeable_ranks=enc_base._mergeable_ranks,
                special_tokens={
                    **enc_base._special_tokens,
                    "<|endofprompt|>": enc_base.max_token_value + 1,
                    "<|padding|>": enc_base.max_token_value + 2,
                    "\n\nSystem: ": enc_base.max_token_value + 3,
                    "\n\nHuman: ": enc_base.max_token_value + 4,
                    "\n\nAssistant: ": enc_base.max_token_value + 5,
                }
            )

        self.eot_text: str = "<|endoftext|>"      # End-of-text special token.
        self.eop_text: str = "<|endofprompt|>"    # End-of-prompt special token.
        self.pad_text: str = "<|padding|>"        # <|padding|>` instead...

        self.sys_text: str = "\n\nSystem: "
        self.usr_text: str = "\n\nHuman: "
        self.bot_text: str = "\n\nAssistant: "

        self.eot_token: int = self.enc.encode(self.eot_text, allowed_special={self.eot_text})[0]
        self.eop_token: int = self.enc.encode(self.eop_text, allowed_special={self.eop_text})[0]
        self.pad_token: int = self.enc.encode(self.pad_text, allowed_special={self.pad_text})[0]

    def train(self, document: str) -> None:
        pass

    def encode(self, text: str) -> List[int]:
        """
        Encode text using the tokenizer's encoding.

        Args:
            text (str): The input text to be encoded.

        Returns:
            List[int]: The list of encoded tokens as integers.
        """
        return self.enc.encode(text, allowed_special={self.eot_text, self.eop_text, self.sys_text, self.usr_text, self.bot_text})

    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens back into text using the tokenizer's decoding.

        Args:
            tokens (List[int]): The list of tokens to be decoded.

        Returns:
            str: The decoded text.
        """
        try:
            return self.enc.decode(tokens)
        except Exception as e:
            print(f"Warning: Failed to decode tokens: {e}")  # Warning if decoding fails.
            return ""
        except BaseException as e:
            print(f"Critical: Failed to decode tokens: {e}")  # Critical error if decoding fails.
            return ""

    def vocab_size(self) -> int:
        """
        Get the vocabulary size of the tokenizer.

        Returns:
            int: The vocabulary size.
        """
        return self.enc.n_vocab
