# Data Analysis Agent — README

A CLI-based data analysis assistant that answers business questions over **BigQuery’s** public dataset `bigquery-public-data.thelook_ecommerce` using an **agentic workflow**.

---

## Project Structure

```
.
├── app.py                       
├── logs/
│   └── app.log                  
├── data_analysis_agent/
│   ├── agent.py                 
│   ├── state.py                 
│   └── files/
│       ├── system_prompts.json  
│       └── config.json          
├── sql_agent/
│   ├── agent.py                 
│   ├── state.py
│   ├── bq_client.py             
│   └── files/
│       └── system_prompts.json
│       └── config.json          
├── plot_agent/
│   ├── agent.py                 
│   ├── state.py
│   ├── plots/                   
│   └── files/
│       └── system_prompts.json
│       └── config.json          
└── helper_functions.py          
```

---
#### Data Flow
1) **User question** enters **Supervisor**.  
2) If **explore**, **Explorer** emits SQL questions + plot description.  
3) **SQL Agent** answers each question by:
   - Generating one BigQuery SQL  
   - Executing it and returning results  
4) **Plot Agent**:
   - Fetches data via its own SQL step  
   - Generates & executes a Matplotlib script  
   - Saves PNG.
   - create plot analysis text
5) **Final Answer Generator** consolidates all evidence → final answer.

#### Agent’s diagrams
Each agent’s diagram is included in its own directory.

#### Error Handling & Fallbacks 
For every SQL or Python execution, we wrap the run in a try/except.
If an error occurs, we log it and feed the full context back into the LLM to attempt an automatic fix.
Each SQL/Python step has a configurable maximum retry count (default: 3) in the agent’s config file.
Each SQL query is validated by scanning for disallowed commands before execution.

#### Reasoning for chosen Cloud services and LLM models
Each agent (SQL and Plot) uses two LLM handles: llm and sota_llm.
The sota_llm is reserved for critical steps—primarily SQL/code generation—to maximize correctness and robustness.
The standard llm is used for all other tasks (analysis, narration, error explanations) to balance quality and cost.
For the Plot agent’s plot analysis step, the selected model supports multimodal input (text + image),
allowing it to interpret the generated chart image alongside the question context for more accurate and grounded commentary.
---

## 1) Setup

### Prerequisites
- Python 3.10+ (tested on 3.12)
- Google Cloud project with access to `bigquery-public-data`
- A Google **service account** JSON credentials file
- An LLM API key (Google Gemini or your configured provider)

### Install
```bash
# clone your repo, then:
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Environment
Create a `.env` file in the project root.

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service_account.json"
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```


---

## License

MIT (or your chosen license)

---
