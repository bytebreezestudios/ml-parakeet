# Parakeet: The Tiny LLM / The Most Premium of Parakeets

## Introduction

Parakeet is a tiny language model designed from scratch for the purpose of research and understanding.

It was trained on an NVIDIA 3080 Ti (12GB) over the course of a few months on a dataset currently weighing in at 50GB. In comparison, the model file itself is 1.5GB.

It's designed to answer the following questions:
- How small can a language model be and still be useful?
- What happens if we over-saturate a tiny model with a massive amount of data?
- Do language models actually generate responses by learning, or do they just memorise and regurgitate?

It is an 18 layer, 18 head, 378M parameter model capable of generating coherent responses to a variety of prompts.

Considering the current limitations, it is not recommended to use this model for any serious applications, however it is very accessible due to its small size and can be used for educational purposes.

## Installation

This is an initial release for the sake of sharing the project and acquiring feedback (training instructions and data will be added in the very near future depending on interest).

1. To install dependencies, please run the `./00-install.sh` script or open it with an editor and run the commands manually.
2. To download the model, grab a copy from [This Google Drive Link](https://drive.google.com/file/d/14GG7AhKQviMcYDgQcnp9P9YAXNYzV6gf/view?usp=drive_link)
   1. Copy the model to the `/assets/models` directory.

## Usage

Simply run the `main.py` script. If everything went well, you should see the following prompt:

```txt
Human: <write your input here then press enter>
```

## Contributing

Found an error? Want to add a feature? Please open an issue or submit a pull request. Your feedback is welcome!

## Collaboration

I plucked Parakeet out of its Jupyter Notebook and created this repository over the course of a few hours:
- The folks at the [Virtual Valley AI](https://discord.gg/cgczW4zu7N) Discord showed interest due to the educational aspect of the project.
- I can be found at the [Byte Breeze Studios](https://discord.gg/7MnbysEyAG) Discord under `Jonno` / `razodactyl`
- This is a 10% project, so I'm not able to dedicate too much time, although I'm absolutely happy to teach and learn from others.

## Roadmap

- [x] Transcribe out of Jupyter Notebook and share on GitHub.
- [ ] Transcribe training scripts out of Jupyter Notebook.
- [ ] Add scripts to download latest models and datasets automatically.
- [ ] Write a detailed blog post on the project.

## References / Special Thanks / (Not Exhaustive)

- NanoGPT by Andrej Karpathy - https://github.com/karpathy/nanoGPT, https://www.youtube.com/watch?v=kCc8FmEb1nY
- ALiBi / Train Short Test Long - https://arxiv.org/pdf/2108.12409
- Bits and Bytes - https://github.com/bitsandbytes-foundation/bitsandbytes
- ...others as referenced in the code itself.

## License

This project is licensed under the GNU General Public License in its initial release. We may change this in the future depending on the direction of the project. Feedback is welcome.
