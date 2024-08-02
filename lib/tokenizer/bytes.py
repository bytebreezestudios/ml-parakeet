from typing import List, Dict
from base import BaseTokenizer


class ByteTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()
        # Pre-define the byte values for special tokens
        self.special_tokens: Dict[str, int] = {
            "<|endoftext|>": 0,
            "<|endofprompt|>": 1,
            "<|padding|>": 2,
            "\n\nSystem: ": 3,
            "\n\nHuman: ": 4,
            "\n\nAssistant: ": 5
        }
        # Reverse mapping for decoding
        self.byte_to_special: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        # Start indices after special tokens
        self.current_index = max(self.special_tokens.values()) + 1

        self.eot_text: str = "<|endoftext|>"      # End-of-text special token.
        self.eop_text: str = "<|endofprompt|>"    # End-of-prompt special token.
        self.pad_text: str = "<|padding|>"        # <|padding|>` instead...

        self.sys_text: str = "\n\nSystem: "
        self.usr_text: str = "\n\nHuman: "
        self.bot_text: str = "\n\nAssistant: "

        self.eot_token: int = self.encode(self.eot_text)[0]
        self.eop_token: int = self.encode(self.eop_text)[0]
        self.pad_token: int = self.encode(self.pad_text)[0]

    def train(self, document: str) -> None:
        # ByteTokenizer does not require training as it works on byte level.
        pass

    # def encode(self, text: str) -> List[int]:
    #     tokens = []
    #     index = 0
    #     while index < len(text):
    #         match = None
    #         # Check if the upcoming sequence matches any special token
    #         for token in self.special_tokens:
    #             if text.startswith(token, index):
    #                 match = token
    #                 break
    #         if match:
    #             # We found a special token, so append its value and skip its length in the text
    #             tokens.append(self.special_tokens[match])
    #             index += len(match)
    #         else:
    #             # No special token found, encode the current character as bytes
    #             tokens.append(ord(text[index]))
    #             index += 1
    #     return tokens

    def encode(self, text: str) -> List[int]:
        tokens = []
        buffer = ""
        for char in text:
            buffer += char
            # Check for special tokens in the buffer
            if buffer in self.special_tokens:
                tokens.append(self.special_tokens[buffer])
                buffer = ""  # Clear buffer after finding special token
            else:
                # Encode buffer contents if no more matches can be found
                potential_match = any(token.startswith(buffer) for token in self.special_tokens)
                if not potential_match:
                    tokens.extend(buffer.encode('utf-8'))
                    buffer = ""  # Clear buffer when no special token is detected
        # Handle any remaining characters in the buffer
        if buffer:
            tokens.extend(buffer.encode('utf-8'))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        # Decode a list of tokens back into a string
        bytes_list = bytearray()
        for token in tokens:
            if token in self.byte_to_special:
                bytes_list.extend(self.byte_to_special[token].encode('utf-8'))
            else:
                bytes_list.append(token)
        return bytes_list.decode('utf-8', errors='replace')

    def vocab_size(self) -> int:
        # Vocab size is 256 (0-255) plus the number of special tokens. (Added 32 for any extensions).
        return 256 + len(self.special_tokens) + 32

if __name__ == "__main__":
    enc_text = "\n\nSystem: Hello, World! This is an example. å∫çABC123!@# 😉😉😉<|endofprompt|>"

    # Example usage:
    tokenizer = ByteTokenizer()
    encoded_text = tokenizer.encode(enc_text)
    decoded_text = tokenizer.decode(encoded_text)

    print(f"Encoded Text ({len(encoded_text)} tok.): {encoded_text}")
    print(f"Decoded Text: {decoded_text}")

    tokenizer = TikTokenizer()
    encoded_text = tokenizer.encode(enc_text)
    decoded_text = tokenizer.decode(encoded_text)

    print(f"Encoded Text ({len(encoded_text)} tok.): {encoded_text}")
    print(f"Decoded Text: {decoded_text}")

    tokenizer = TikTokenizer(encoding="gpt2")
    encoded_text = tokenizer.encode(enc_text)
    decoded_text = tokenizer.decode(encoded_text)

    print(f"Encoded Text ({len(encoded_text)} tok.): {encoded_text}")
    print(f"Decoded Text: {decoded_text}")
