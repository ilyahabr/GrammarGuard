import os

import torch
from langchain import HuggingFacePipeline, PromptTemplate
from optimum.bettertransformer import BetterTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# WHISPER

whisper = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-large-v2",
    torch_dtype=torch.float16,
    device=DEVICE,
)
whisper.model = BetterTransformer.transform(whisper.model)
whisper.model.eval()

# LAMMA V2

MODEL_NAME = "TheBloke/Llama-2-13b-Chat-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

lammav2 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map=DEVICE,
)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

text_pipeline = pipeline(
    "text-generation",
    model=lammav2,
    tokenizer=tokenizer,
    return_full_text=True,
    generation_config=generation_config,
)


def speech2text(audio, whisper):
    transcription = whisper(audio, chunk_length_s=30, stride_length_s=5, batch_size=8)
    return transcription["text"]


def congifure_llm():
    template = """
                    [INST] <>
                    Act as a native english speacker who is teaching english to university students. 
                    You should fix grammer and explain shortly how to say same idea grammarly right. 
                    Explaine to him what tense he used and what tense he should use.
                    <>

                    {text} [/INST]
                    """

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )
    return prompt


if __name__ == "__main__":

    audio = "86cb9188-3db4-41d2-a939-ec5c95cbeff1.webm"
    text = speech2text(audio, whisper)

    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})
    prompt = congifure_llm()

    result = llm(prompt.format(text=text))
