from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

def build_qa_chain(retriever, use_gpu=False):
    template = """
You are a helpful assistant. Use ONLY the context to answer.
If answer is not in context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=template
    )

    gen = get_generation_pipeline(use_gpu=use_gpu)
    llm = make_llm(gen)

    chain = (
        RunnableParallel({
            "context": retriever,
            "query": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    return chain
