def get_map_prompt(batch_text, analysis):
    mode = analysis["mode"]

    if mode == "qa":
        instruction = "Extract only factual answers relevant to the question."

    elif mode == "generation":
        instruction = "Extract structured points that can help generate content (MCQs, summaries, etc.)."

    elif mode == "research":
        instruction = "Extract deep insights, explanations, and relationships between concepts."

    else:
        instruction = "Extract relevant information."

    return f"""
You are analyzing a section of a document.

Task: {analysis['Query']}
Mode: {mode}

Instructions:
{instruction}

Section:
{batch_text}

Return only relevant information.
"""
    
#-----------------------------------------------------------------------------------------------------------------------

async def reduce_step(partials, analysis, llm):

    combined = "\n\n".join(partials)
    mode = analysis["mode"]

    prompt = f"""
You are a senior AI system combining multiple partial analyses.

Task: {analysis['Query']}
Mode: {mode}

Partial Results:
{combined}

Instructions:
- Remove redundancy
- Merge similar ideas
- Ensure logical flow
- Produce a final high-quality response

If mode is:
- QA → give precise final answer
- generation → structured output (MCQs / list / etc.)
- research → deep structured explanation

Final Answer:
"""

    response = await llm.ainvoke(prompt)
    return response.content

#-----------------------------------------------------------------------------------------------------------------------

async def map_reduce_pipeline(chunks, analysis, llm, file_name):

    BATCH_SIZE = 5

    batched_chunks = [
        chunks[i:i+BATCH_SIZE]
        for i in range(0, len(chunks), BATCH_SIZE)
    ]

    partials = []

    for batch in batched_chunks:
        batch_text = " ".join(batch)

        prompt = get_map_prompt(batch_text, analysis)

        response = await llm.ainvoke(prompt)
        partials.append(response.content)

    final_answer = await reduce_step(partials, analysis, llm)

    return final_answer