from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate


########################## Prompt ##########################

map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""

combine_paragraph_prompt = """
Your Role: Efficient Summarizer
Short basic instruction: Summarize the provided text into an essay format.
What you should do: You are tasked with digesting the text enclosed within triple backquotes, focusing intently on distilling its primary themes, key points, and essential details. Your summary should be crafted into an essay format comprising an introduction, body (detail), and conclusion (summary), adhering to the criteria outlined below:
Introduction: Open with a clear, engaging statement that introduces the overall theme or central idea of the text. Provide a brief overview that piques interest and sets the stage for the detailed summary.
Detail (Body): Elaborate on the critical aspects and main points from the text. Integrate these elements seamlessly, ensuring a coherent and focused narrative that highlights the text's essence without unnecessary detail.
Summary (Conclusion): Conclude with a succinct recapitulation of the text's core message, weaving together the significant points discussed in the body to reinforce the central theme.
Your Goal: To provide a clear, concise, and comprehensive summary that captures the essence of the text, offering readers a coherent overview of its most crucial elements in a brief essay format.
Result: An essay-formatted summary consisting of an introduction, body, and conclusion, capturing the text's primary themes and key points concisely and effectively.
Constraint: Keep the summary concise, aiming to capture the text's essence without exceeding a few sentences for each section (introduction, detail, and summary).
Context: The text for summarization is presented within triple backquotes. Focus on extracting and condensing the primary information, offering a narrative that reflects the core message of the text.
```{text}```
PARAGRAPH SUMMARY:
"""

combine_bullet_prompt = """
Your Role: Content Summarizer
Short basic instruction: Summarize the provided text into bullet points.
What you should do: Analyze the given text enclosed within triple backquotes and distill its essence into a succinct, bullet-point format summary.
Your Goal: To create a clear and focused summary that captures the most critical aspects of the text, organized as 3-5 bullet points.
Result: A bullet-point summary, with each point encapsulating a significant theme or key information from the text. The summary should be comprehensive yet concise, highlighting only the essential elements.
Constraint: Limit the summary to 3-5 bullet points, focusing solely on the most crucial aspects of the text. Ensure clarity and brevity in each bullet point.
Context: The text to be summarized is provided without specific context regarding its nature (technical, literary, etc.). The emphasis is on distilling primary information into a clear overview.
```{text}```
BULLET POINT SUMMARY:
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
combine_paragraph_prompt_template = PromptTemplate(template=combine_paragraph_prompt, input_variables=["text"])
combine_bullet_prompt_template = PromptTemplate(template=combine_bullet_prompt, input_variables=["text"])

########################## Map-reduce ##########################

def map_reduce_paragraph(docs, model):
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce',
                                         map_prompt=map_prompt_template,combine_prompt=combine_paragraph_prompt_template,
                                         # verbose=True
                                         )
    output = summary_chain.run(docs)
    return output

def map_reduce_bullet(docs, model):
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce',
                                         map_prompt=map_prompt_template,combine_prompt=combine_bullet_prompt_template,
                                         # verbose=True
                                         )
    output = summary_chain.run(docs)
    return output

########################## Refine ##########################

def refine_paragraph(docs, model):
    chain = load_summarize_chain(llm=model, chain_type="refine",
                                 question_prompt=map_prompt_template,
                                 refine_prompt=combine_paragraph_prompt_template,
                                 return_intermediate_steps=False,
                                 input_key="input_documents", output_key="output_text",
                                 )
    output = chain({"input_documents": docs}, return_only_outputs=True)
    return output.get("output_text", "")

def refine_bullet(docs, model):
    chain = load_summarize_chain(llm=model, chain_type="refine",
                                 question_prompt=map_prompt_template,
                                 refine_prompt=combine_bullet_prompt_template,
                                 return_intermediate_steps=False,
                                 input_key="input_documents", output_key="output_text",
                                 )
    output = chain({"input_documents": docs}, return_only_outputs=True)
    return output.get("output_text", "")

########################## Translate ##########################

template_string = """
Your task is to translate the text found between the triple backticks below into Thai language.
Ensure the translation maintains a natural and fluent tone, and exclude the backticks from your response. Below is the text requiring translation:
```{text}```
Please adhere to the following guidelines in your translation:
- The translation should be natural and fluent, accurately reflecting the essence of the original text.
- Focus on translating only the section enclosed within the triple backticks; all other instructions should remain in English.
- Exclude the backticks in your translation, presenting a clear and focused response.
- Keep the format consistent: if the original text is in bullet points, maintain bullet points; if it's in paragraph form, keep it as a paragraph.
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

def translate_to_thai(docs, model):
    prompt = prompt_template.format_messages(text=docs)
    output = model(prompt)
    return output.content