# AI Consult
EECE 490 Project on agentic workflow

AI Consult is a fully automated agentic worflow designed to bridge the gap between non technical users (including companies) and AI solutions: 0 code experience required!
Our agents take the user from raw data to a trained Machine Learning model seamlessly. 

Just upload your CSV files, and let AI Consult analyze, recommend, and implement the best-fit ML model to solve your real-world problem.

## Prerequisites
 - Docker

## Installation
Clone the repository
```bash
git clone https://github.com/HadiMchawrab/Agents.git
cd Agents
```
Create a .env file inside the backend directory, with CLAUDE_API_KEY, LANGSMITH_API_KEY, and OPENAI_API_KEY.

Start docker
```bash
docker-compose up --build
```

## Workflow
_The pipeline communicates with an LLM by sending well designed prompts and receiving formatted responses that are forwared from stage to stage._

1. Upload CSV File
   Start by uploading your data file(s).

2. Topic Inference
   The system analyzes the metadata to find interesting topics and ML models that could be used on this data.
   The agent then does some web scraping to find real life applied models in each topic. This could take up to 25 minutes. You can check the logging to see scraping progress (selenium container).

4. Human in the loop
   The topics scraped are displayed to the user, each with the reasoning behind it, the relationship between the data and the models, the data needs for a scpecific task/model, and a list of possible ML models to be applied.
   The user selects one topic to move forward with, and has the possibility of adding extra data to the analysis.

5. Automated Data Analysis
   The system analyzes data structure & types, and visualizes distributions, correlations, and patterns.
   This is done by generating python scripts taking into account the type and shape of the data, and then executing them in an isolated jupyter notebook.
   The resulting images are sent to the next stage of the pipeline: ML model choice.

6. ML Model Recommendation
   The system chooses the best-fit ML model(s) based on data characteristics, and generates a training and/or running script of the model on the selected data.

8. Model Training & Evaluation
   The selected model is trained on your data, and key performance metrics are shared.
   The final results are currently shown in the terminal.

## Further adjusments
- Training results displayed in the frontend
- Creating a loop between performance metrics and the executable script for model optimization.
