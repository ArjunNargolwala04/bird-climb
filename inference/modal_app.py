"""
Modal app serving Qwen2.5-Coder-32B-Instruct via vLLM.

Deploy:  modal deploy inference/modal_app.py
Test:    modal run inference/modal_app.py

The app exposes a single class (Qwen32B) with .generate() and
.generate_batch() methods callable remotely from generate.py.
"""

import modal

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
VOLUME_NAME = "bird-climb-models"

app = modal.App("bird-climb")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.8.0",
        "transformers",
        "torch",
    )
)


@app.cls(
    image=vllm_image,
    gpu="B200",
    volumes={"/models": volume},
    timeout=600,
    scaledown_window=300,
)
class Qwen32B:
    model_id: str = MODEL_ID

    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.model_id,
            download_dir="/models",
            max_model_len=8192,
            trust_remote_code=True,
            dtype="auto",
        )
        self.tokenizer = self.llm.get_tokenizer()

    @modal.method()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        n: int = 1,
        stop: list[str] | None = None,
    ) -> list[str]:
        """
        Generate SQL completions.

        Returns:
            List of generated strings (length = n)
        """
        from vllm import SamplingParams

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=stop or [],
        )

        outputs = self.llm.generate([prompt_text], params)
        return [out.text.strip() for out in outputs[0].outputs]

    @modal.method()
    def generate_batch(
        self,
        prompts: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        n: int = 1,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """
        Batched generation for throughput. Each prompt dict has
        'system' and 'user' keys.

        Returns:
            List of lists — outer[i] has n completions for prompt i.
        """
        from vllm import SamplingParams

        prompt_texts = []
        for p in prompts:
            messages = [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ]
            prompt_texts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=stop or [],
        )

        outputs = self.llm.generate(prompt_texts, params)
        return [
            [out.text.strip() for out in output.outputs]
            for output in outputs
        ]


@app.local_entrypoint()
def main():
    """Quick smoke test."""
    model = Qwen32B()
    results = model.generate.remote(
        system_prompt="You are an expert SQL developer. Return only SQL, no explanation.",
        user_prompt="Write a SQL query to count the number of rows in a table called 'users'.",
        max_tokens=128,
        temperature=0.0,
    )
    print(f"Generated {len(results)} completion(s):")
    for i, r in enumerate(results):
        print(f"  [{i}] {r}")
