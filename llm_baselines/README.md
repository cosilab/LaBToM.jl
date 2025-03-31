# LLM Baselines for Epistemic Language Understanding

All LLM baselines are implemented in `llm_baselines.py`. To run the Gemini Pro baselines, the [`google.generativeai`](https://github.com/google-gemini/deprecated-generative-ai-python/tree/v0.8.4?tab=readme-ov-file) package is required. Users should specify their OpenAI and Gemini API keys at the top of `llm_baselines.py` (ideally by specifying the environment variables `OPENAI_API_KEY` and `GEMINI_API_KEY`).

Note that the the few-shot prompted LLM baselines require access to human-provided ratings. See the main README for how human ratings can be downloaded.
