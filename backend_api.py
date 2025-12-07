from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "google/flan-t5-small"

def get_generation_pipeline(model_name=MODEL, use_gpu=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if use_gpu else -1
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256
    )

def make_llm(gen_pipeline):
    return HuggingFacePipeline(pipeline=gen_pipeline)

def build_qa_chain(retriever, use_gpu=False):
    gen = get_generation_pipeline(use_gpu=use_gpu)
    llm = make_llm(gen)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain
