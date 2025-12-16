import json
import httpx
import transformers

vllm_endpoint = "http://127.0.0.1:59000"

request = [
    {
        "content": 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nA function $f$ is defined on integers such that:\n\n- $f(n) = n + 3$ if $n$ is odd.\n- $f(n) = \\frac{n}{2}$ if $n$ is even.\n\nIf $k$ is an odd integer, determine the values for which $f(f(f(k))) = k$.\n\nRemember to put your answer on its own line after "Answer:".',
        "role": "user",
    }
]


lora_name = "069410cf-7c97-7a22-8000-cb3c9921fc2a"

cli = httpx.Client(timeout=3600.0)
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
prompt: str = tokenizer.apply_chat_template(
    request,
    add_generation_token=True,
    tokenize=False,
)
request_json = {"model": "base", "prompt": prompt, "max_tokens": 16, "seed": 42}
resp = cli.post(f"{vllm_endpoint}/v1/completions", json=request_json)
resp.raise_for_status()
print(resp.json())

request_json["model"] = lora_name
resp = cli.post(f"{vllm_endpoint}/v1/completions", json=request_json)
resp.raise_for_status()
print(resp.json())
