AGENTIC AI SYSTEM FOR AUTOMATION OF SEMICONDUCTOR DEFECT ANALYSIS 
  
  The goal of this project is to develop an Agentic AI system that predicts semiconductor defects and automatically identifies their root causes, significantly 
  reducing the time and effort required for manual defect analysis and process troubleshooting.
  
  Currently, the system focuses on the three key processes that most impact wafer yield—deposition, etching, and lithography, and can be extended to other semiconductor 
  fabrication processes for comprehensive defect prediction and root cause analysis.
  
  The AI system follows a modular architecture designed for intelligent reasoning and workflow automation:

    1) LLM (OpenAI): Provides reasoning and planning capabilities
    2) LangChain / LangGraph: Orchestrates the overall workflow
    3) Pinecone: Retrieves relevant memory and contextual information
    4) Python: Implements defect prediction and automated root cause analysis

 The core implementation resides in the root_cause_suggestion.py file. It leverages XGBoost models to predict wafer defects and joining issues, while SHAP values 
 provide interpretable insights into the most influential features contributing to potential defects. The orchestrator maps these top features to specific
 fabrication processes—lithography, deposition, and etching—generating actionable recommendations for process engineers.

 Model performance has been validated on test data, demonstrating strong predictive capability for defect and join failure detection, making the system directly 
 applicable in semiconductor manufacturing for proactive defect management and yield improvement.

 Currently, the project is still a work in progress. A key area of future development is extending the Agentic AI system to automate CAPA (Corrective and Preventive Action) 
 solutions, which would require integration with real-time process data and broader fab-wide datasets. Additionally, the system could be enhanced to cover more 
 fabrication processes beyond deposition, etching, and lithography, improve model explainability, and incorporate dynamic feedback loops to continuously optimize 
 defect prediction and process yield. 
 
    
    
