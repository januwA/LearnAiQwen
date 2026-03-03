
查看 Qwen 组织的模型

```sh
❯ uv run hf models ls --author Qwen
ID                                  CREATED_AT DOWNLOADS LIKES LIBRARY_NAME TAGS                                PIPELINE_TAG       TRENDING_SCORE
----------------------------------- ---------- --------- ----- ------------ ----------------------------------- ------------------ --------------
Qwen/Qwen3.5-35B-A3B                2026-02-24 258764    594   transformers transformers, safetensors, qwen3... image-text-to-text 594
Qwen/Qwen3.5-27B                    2026-02-24 107964    388   transformers transformers, safetensors, qwen3... image-text-to-text 388
Qwen/Qwen3.5-397B-A17B              2026-02-16 725954    1105  transformers transformers, safetensors, qwen3... image-text-to-text 364
Qwen/Qwen3.5-122B-A10B              2026-02-24 107821    320   transformers transformers, safetensors, qwen3... image-text-to-text 320
Qwen/Qwen3-TTS-12Hz-1.7B-CustomV... 2026-01-21 1071196   1206               safetensors, qwen3_tts, text-to-... text-to-speech     108
Qwen/Qwen3-Coder-Next               2026-01-30 685871    1015  transformers transformers, safetensors, qwen3... text-generation    95
Qwen/Qwen3.5-35B-A3B-Base           2026-02-24 2888      73    transformers transformers, safetensors, qwen3... image-text-to-text 73
Qwen/Qwen3.5-397B-A17B-FP8          2026-02-18 188096    108   transformers transformers, safetensors, qwen3... image-text-to-text 72
Qwen/Qwen3.5-35B-A3B-FP8            2026-02-25 52778     40    transformers transformers, safetensors, qwen3... image-text-to-text 40
Qwen/Qwen3.5-27B-FP8                2026-02-25 24765     39    transformers transformers, safetensors, qwen3... image-text-to-text 39
```


```
Qwen3.5-35B-A3B 的含义：
35B = 总参数量 350亿（350亿个参数存储在硬盘上）
A3B = 激活参数 30亿（A = Activated，运行时实际加载到显存的参数量）
```

下载模型

```shell
uv run hf download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir ./qwen2.5-3b
```

启动 OpenAI-compatible 服务（对外暴露接口，内部走本地权重）

```shell
uv run ai-qwen-openai-api ^
  --model-path .\qwen2.5-3b ^
  --served-model-name qwen-local ^
  --host 0.0.0.0 ^
  --port 8000
```

OpenAI SDK 调用示例（客户端）：

```shell
python - <<'PY'
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="qwen-local",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
)
print(resp.choices[0].message.content)
PY
```
