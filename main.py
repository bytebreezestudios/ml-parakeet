import os
import gc

import torch

from lib.chat_history import ChatHistory
from lib.tokenizer.tiktoken import TikTokenizer
from model import ParakeetGPT, ParakeetConfig


if __name__ == "__main__":
    model = None
    device = None
    optimizer = None
    scaler = None

    N_STEPS_PER_TQDM_UPDATE = 10
    N_STEPS_PER_EVALUATION  = 5000
    N_STEPS_PER_CHECKPOINT  = 10000

    # tokenizer = TikTokenizer(encoding="gpt2")  # `gpt2` tokenizer.pad_token = 50258 (ignore_index=50258)
    tokenizer = TikTokenizer(encoding="cl100k_base")  # `gpt3` tokenizer.pad_token = 100277 (ignore_index=100277)

    # TODO:
    # - N_STEPS_PER_CHECKPOINT_BACKUP = 50000
    # - Checkpoint backup should contain today's date, ignore time as we don't want to fill up our SSD.
    # TODO: Save every N minutes, not steps.
    # TODO: Config file to alter settings during training / not have to change settings in code.
    # TODO: In config: Switch between validation display or example datasets.


    def initialize_model(tokenizer, v_nearest64=None):
        """
        Initialize the model and move it to GPU if available. If GPU fails, fallback to CPU.

        Returns:
            torch.nn.Module: Initialized model.
            torch.device: The device on which the model is located (GPU or CPU).
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with torch.no_grad():
                torch.cuda.empty_cache()  # Clear any previously cached allocations on the GPU.
                gc.collect()

            if device == torch.device("cuda"):
                # Test if GPU is available by trying to allocate a small tensor on it.
                test_tensor = torch.tensor([1], device=device)
                del test_tensor  # Delete the test tensor after checking to free up GPU memory.
        except Exception as e:
            # GPU failed, fallback to CPU.
            print(f"GPU failed to initialize: {e}")
            device = torch.device("cpu")

        if v_nearest64 is None:
            v_nearest64 = (tokenizer.vocab_size() + (64 - (tokenizer.vocab_size() % 64)))

        model = ParakeetGPT(
            ParakeetConfig(
                vocab_size=v_nearest64,
                n_layer=18,
                n_head=18,
                n_embd=1152,
                n_fexp=4,
                block_size=8192,
                dropout=0.07,
                bias=False,
                gqa=True,
                n_blocks_per_kv=2,
                kv_cache=True,
                name="parakeet8k",
            )
        )

        # Print the model parameters to showcase your magnificent creation.
        print(model)

        return model, device

    # Now, let's use the function to initialize the model and get the selected device:
    # - If it's "cuda", you're ready to rev up those fryers! If not, well, the slow roast begins!
    if not model:
        model, device = initialize_model(tokenizer)

        # Device selected, transfer model across (Unless running inference)
        model = model.to(device)

        # TODO: Model-class itself should handle saving/file-naming etc.
        filename = f"./assets/models/{model.config.name}-c{model.config.block_size}{'b' if model.config.bias else ''}-d{model.config.n_embd}-fexp{model.config.n_fexp}_v{model.config.vocab_size}_h{model.config.n_head}_l{model.config.n_layer}-alibi{f'+gqa{model.config.n_blocks_per_kv}' if model.config.gqa else ''}.pth"

        if os.path.isfile(filename):
            model.load_state_dict(torch.load(filename, map_location=device))

    chat = ChatHistory(max_history=8)

    chat.add_message(role="Human", content="You're Parakeet. An AI by Byte Breeze Studios. You are a large language model AI designed in Brisbane. You speak in full sentences and always try to be helpful to your users.")

    for i in range(40):
        query = input("\nHuman: ")

        chat.add_message(role="Human", content=query)

        conversation = chat.transcript()

        print("Assistant: ", end='')

        response = model.generate(
            device,
            tokenizer,
            f"{conversation}\n\nAssistant: ",
            max_length=4000,
            freq_penalty=0.07,
            pres_penalty=0.02,
            temperature=0.30,
            top_k=-1,
            top_p=0.95,
            min_p=0.01,
            greedy=False,
            token_callback=lambda t: print(t, end='')
        )

        chat.add_message(role="Assistant", content=response)

    print(chat.transcript())
