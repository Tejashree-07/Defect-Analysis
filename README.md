AGENTIC AI SYSTEM FOR AUTOMATION OF SEMICONDUCTOR DEFECT ANALYSIS 
  
The goal of this project is to develop an Agentic AI system that predicts semiconductor defects and automatically identifies their root causes, significantly reducing the time and effort required for manual defect analysis and process troubleshooting.
  
Currently, the system focuses on the three key processes that most impact wafer yield - deposition, etching, and lithography, and can be extended to other semiconductor fabrication processes for comprehensive defect prediction and root cause analysis. The system uses a dataset from Kaggle that includes inline tool and process parameters, and ategorized yes/no kind of data 
  
The AI system follows a modular architecture designed for intelligent reasoning and workflow automation:

    1) LLM (OpenAI): Provides reasoning and planning capabilities
    2) Pinecone: Retrieves relevant memory and contextual information
    3) Python: Implements defect prediction and automated root cause analysis
    4) Digital Twin Simulation: Generates realistic semiconductor manufacturing data streams for continuous system validation and testing

The core implementation resides in the agentic_AI_defect_analysis.py file. It leverages XGBoost models to predict wafer defects and joining issues, while SHAP values provide interpretable insights into the most influential features contributing to potential defects. The orchestrator maps these top features to specific fabrication processes, lithography, deposition, and etching—generating actionable recommendations for process engineers.

A key component of this system is the semiconductor Digital Twin, implemented in digital_twin_simulator.py. The Digital Twin is calibrated on approximately 42,000 historical wafers from the dataset, learning the statistical distributions and inter-parameter correlations across all process sensors. Once calibrated, it generates a continuous stream of synthetic wafers that preserve the underlying relationships observed in real manufacturing data. This enables realistic simulation of fab behavior, including gradual sensor drift in either direction, and transient equipment faults such as temperature spikes, RF power drops, vacuum loss, particle bursts, and vibration events without requiring live production data. As a result, the system can be used to test and validate defect detection, predictive maintenance, and root cause analysis workflows under controlled yet realistic conditions.
 
 Currently, the project is still a work in progress. A key area of future development is extending the Agentic AI system to automate CAPA (Corrective and Preventive 
 Action) workflows, which would require integration with real-time process control data and broader fab-level datasets. Additional enhancements include temporal 
 sequence modeling, real-time streaming integration with manufacturing systems (MES/FDC/SPC), expanded fabrication process coverage, and adaptive learning mechanisms 
 that continuously improve defect prediction and process optimization over time. 
 
    
    
