
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate


# Map reduce
map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""

combine_prompt = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
It should be only 3-5 bullet points so pick only important points.
```{text}```
BULLET POINT SUMMARY:
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

def map_reduce_result(docs, model):
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce',
                                         map_prompt=map_prompt_template,combine_prompt=combine_prompt_template,
                                         # verbose=True
                                         )
    output = summary_chain.run(docs)
    return output


# Refine
prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Return your response in bullet points which covers the key points of the text."
    "It should be only 3-5 bullet points so pick only important points."
    "If the context isn't useful, return the original summary."
)

refine_prompt = PromptTemplate.from_template(refine_template)
prompt = PromptTemplate.from_template(prompt_template)


def refine_result(docs, model):
    chain = load_summarize_chain(llm=model, chain_type="refine",
                                 question_prompt=prompt, refine_prompt=refine_prompt,
                                 return_intermediate_steps=False,
                                 input_key="input_documents", output_key="output_text",
                                 )
    output = chain({"input_documents": docs}, return_only_outputs=True)
    return output