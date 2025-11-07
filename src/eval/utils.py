
def filter_by_context_length(data, model):
    model = model.split("/")[-1].lower()
    if model.startswith("gpt-5"):
        import tiktoken

        max_context_length = 272000
        tok = tiktoken.get_encoding("o200k_base")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("o4"):
        import tiktoken

        max_context_length = 200000
        tok = tiktoken.get_encoding("o200k_base")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("o3"):
        import tiktoken

        max_context_length = 200000
        tok = tiktoken.get_encoding("o200k_base")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("deepseek-r1-0528"):
        from transformers import AutoTokenizer
        
        max_context_length = 131072  # we leave 32k for the response
        tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")

        def tok_count(x):
            return len(tok.encode(x))
    elif model.startswith("llama-4-maverick"):
        from transformers import AutoTokenizer

        max_context_length = 1000000
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
        )

        def tok_count(x):
            return len(tok.encode(x))
        
    elif model.startswith("claude-sonnet-4-20250514"):
        import anthropic
        from secret import ANTHROPIC_API_KEY

        max_context_length = 1000000
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

        max_context_length = 1000000
        genai.configure(api_key=GEMINI_API_KEY)
        gemini = genai.GenerativeModel(model)

        def tok_count(text):
            response = gemini.count_tokens(text)
            return response.total_tokens

    else:
        msg = f"tokenization not supported for {model}"
        raise ValueError(msg)

    data = data.filter(
        lambda x: tok_count(x["context_window_text"]) <= max_context_length,
        desc=f"filtering instances with more than {max_context_length} tokens",
    )
    return data
