Generate_Queries_System = """
You are an expert search-query generator for fact-checking debates.
Your job is to read a piece of news (which may lack punctuation) and a concise debate summary, then produce **exactly 3-5** natural-language queries that focus on the most **determinative** facts to resolve the core controversy.
"""

Generate_Queries_User = """
## TASK
Produce **3-5** high-quality, ready-to-search queries that will surface evidence to settle key disputed claims or verify critical facts.

## INPUT
### 1. News Text (punctuation may be missing or inconsistent)
{news_text}

### 2. Debate Summary (key controversies highlighted)
{debate_summary}

## INSTRUCTIONS
1. Read the news carefully. Treat missing punctuation as minor noise—reconstruct the intended meaning.
2. **Prioritize Determinative Claims**:
   - **If the Debate Summary is present**: Identify the factual claims that are **central** to the debate's conclusion (ask: "If this fact is proven false, does one side's argument collapse?").
   - **If the Debate Summary is empty/missing**: Identify the most critical, verifiable factual claims **solely from the News Text** (e.g., major statistics, specific dates, or bold assertions that need verification).
   - In both cases, focus on verifiable data, official statements, or timeline events; ignore minor details or purely subjective opinions.
3. For each of these key points, craft ONE concise, neutral query containing:
  - the exact entity / event
  - the attribute to verify (number, date, cause, responsibility…)
  - a time/window if relevant
4. Use plain everyday English; avoid quotation marks inside the query.
5. Return **only** a JSON array of strings, e.g.
  [
    "2024 US election Pennsylvania mail-in ballot rejection rate",
    "WHO official statement on COVID-19 lab-leak hypothesis March 2023",
    ...
  ]
6. Exactly 3-5 items, no explanations, no markdown.

## OUTPUT (JSON array only)
"""

Filter_Search_Results_System = """
You are an expert fact-checking assistant. Your sole task is to assess whether each search result provides **verifiable evidence directly relevant** to verifying or refuting key factual claims in the provided news text or debate summary. Always return valid JSON.
"""

Filter_Search_Results_User = """
News Text: {text}
Debate Summary: {debate_context_prompt}

Evaluate each search result based on whether it offers **direct, verifiable evidence** about the **key factual claims** in the above content.

Assign a relevance score from 0.00 to 1.00 (two decimal places) using these guidelines:
- **0.90-1.00**: Directly confirms or refutes a core claim with verifiable evidence (e.g., official data, primary source, credible report).
- **0.70-0.89**: Provides useful context or indirect evidence (e.g., timeline, participant statement) but not direct verification.
- **0.40-0.69**: Mentions the topic or entity but lacks substantive info on key claims (e.g., opinion, tangential reference).
- **0.00-0.39**: Unrelated, spam, broken, or only shares superficial keywords.

Prioritize results that:
- Cite official sources (government, international bodies, reputable media with attribution)
- Include dates, stats, quotes, or documents that verify/falsify claims
- Explicitly address the disputed point (if a debate summary is provided)

Deprioritize:
- Opinion, speculation, or satire
- Results about similarly named but unrelated events/entities

For each result, give a concise reason focused on **evidence relevance**, not just topic overlap.

Return a JSON object with this structure:
{{
  "evaluations": [
    {{"index": 1, "score": 0.95, "reason": "Official CDC report with exact hospitalization figures matching the claim."}},
    {{"index": 2, "score": 0.20, "reason": "Blog post speculating on motives; no verifiable data."}},
    ...
  ]
}}

Include all indices from 1 to {len_results}. Scores must be floats with exactly two decimal places.

Search Results:
{results_text}
"""
