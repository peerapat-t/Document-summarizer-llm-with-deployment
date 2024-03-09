from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate


########################## Prompt ##########################

map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""

combine_bullet_prompt = """
Your job is to summarize the text enclosed within triple backquotes.
The summary should be presented as bullet points, adhering to the criteria below:
- Limit the summary to 3-5 bullet points to ensure it is succinct yet comprehensive.
- Focus on distilling the essence of the text, highlighting only its most critical aspects.
- Each bullet point should encapsulate a significant theme or key point derived from the text.
- Remember to extract and condense the primary information, offering a clear and focused overview.
```{text}```
BULLET POINT SUMMARY:
"""

combine_paragraph_prompt = """
Your job is to summarize the text enclosed within triple backquotes.
The summary should be written in a singal paragraph, adhering to the criteria below:
- Keep the summary concise, aiming for a length that captures the essence of the text without exceeding a few sentences.
- Focus on identifying and integrating the text's most critical aspects, distilling its primary themes and key points.
- Ensure the summary is clear and focused, providing a comprehensive overview by weaving together the significant elements of the text.
- Remember to extract and condense the primary information, offering a seamless narrative that reflects the core message of the text.
```{text}```
PARAGRAPH SUMMARY:
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
combine_bullet_prompt_template = PromptTemplate(template=combine_bullet_prompt, input_variables=["text"])
combine_paragraph_prompt_template = PromptTemplate(template=combine_paragraph_prompt, input_variables=["text"])

########################## Map-reduce ##########################


def map_reduce_bullet(docs, model):
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce',
                                         map_prompt=map_prompt_template,combine_prompt=combine_bullet_prompt_template,
                                         # verbose=True
                                         )
    output = summary_chain.run(docs)
    return output

def map_reduce_paragraph(docs, model):
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce',
                                         map_prompt=map_prompt_template,combine_prompt=combine_paragraph_prompt_template,
                                         # verbose=True
                                         )
    output = summary_chain.run(docs)
    return output


########################## Refine ##########################

def refine_bullet(docs, model):
    chain = load_summarize_chain(llm=model, chain_type="refine",
                                 question_prompt=map_prompt_template,
                                 refine_prompt=combine_bullet_prompt_template,
                                 return_intermediate_steps=False,
                                 input_key="input_documents", output_key="output_text",
                                 )
    output = chain({"input_documents": docs}, return_only_outputs=True)
    return output.get("output_text", "")


def refine_paragraph(docs, model):
    chain = load_summarize_chain(llm=model, chain_type="refine",
                                 question_prompt=map_prompt_template,
                                 refine_prompt=combine_paragraph_prompt_template,
                                 return_intermediate_steps=False,
                                 input_key="input_documents", output_key="output_text",
                                 )
    output = chain({"input_documents": docs}, return_only_outputs=True)
    return output.get("output_text", "")



########################## Translate ##########################

template_string = """
"Translate the text found between the triple backticks below into Thai.
Make sure to keep the translation natural and fluent, omitting the backticks in your response.
Here is the text that needs translation:
```{text}```
Please focus on translating just the section within the triple backticks,
and let everything else remain in English.
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

def translate_to_thai(docs, model):
    prompt = prompt_template.format_messages(text=docs)
    output = model(prompt)
    return output.content