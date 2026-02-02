from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Create HF pipeline (TEXT generation, not chat)
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
)

# Wrap as LangChain LLM (not ChatModel)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Invoke with a STRING prompt
prompt = "What is the capital of France?"
result = llm.invoke(prompt)

print(result)
