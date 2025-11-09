def compute_context_lengths(data, model):
    model = model.split("/")[-1].lower()
    if model.startswith("gpt-5"):
        import tiktoken

        tok = tiktoken.get_encoding("o200k_base")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("o4"):
        import tiktoken

        tok = tiktoken.get_encoding("o200k_base")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("o3"):
        import tiktoken

        tok = tiktoken.get_encoding("o200k_base")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("deepseek-r1-0528"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("llama-4-maverick"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        )

        def tok_count(x):
            return len(tok.encode(x))

    elif model.startswith("claude-sonnet-4-20250514"):
        import anthropic
        from secret import ANTHROPIC_API_KEY

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        def tok_count(x):
            response = client.messages.count_tokens(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": x}],
            )
            return response.input_tokens
    elif model.startswith("gemini-2.5"):
        import google.generativeai as genai
        from secret import GEMINI_API_KEY

        genai.configure(api_key=GEMINI_API_KEY)
        gemini = genai.GenerativeModel(model)

        def tok_count(text):
            response = gemini.count_tokens(text)
            return response.total_tokens

    else:
        msg = f"tokenization not supported for {model}"
        raise ValueError(msg)

    # add a new column for context_len
    return data.map(
        lambda x: {**x, "context_len": tok_count(x["context_window_text"])},
        desc="add context length column",
    )
