# Generative AI and Agentic AI Projects

This repository contains two projects that I built while working with Generative AI, RAG, and agent-based workflows.

The first project focuses on using an LLM with a RAG pipeline to screen resumes against a job description. The second project uses LangGraph to create a multi-agent workflow for processing insurance claims.

Both projects helped me understand how LLMs can be combined with traditional Python, retrieval systems, and workflow orchestration to build practical AI applications.

---

## Projects

### 1. AI Resume Screening Assistant

The Resume Screening Assistant is designed to make the initial resume screening process easier.

Instead of sending the complete resume directly to the LLM, the system first extracts the resume content, breaks it into smaller sections, stores those sections in a FAISS vector database, and retrieves the parts that are most relevant to the given job description.

The retrieved information is then passed to the LLM, which produces a structured evaluation of the candidate.

### How it works

```text
Resume PDF
    |
    v
Extract Resume Text
    |
    v
Split Text into Smaller Chunks
    |
    v
Create Vector Representations
    |
    v
FAISS Similarity Search
    |
    v
Retrieve Relevant Resume Sections
    |
    v
Compare with Job Description
    |
    v
LLM Evaluation
    |
    v
Structured Candidate Result
```

The resume text is divided into 400-character chunks with an overlap of 50 characters. The system retrieves the four most relevant chunks for the job description before sending the information to the LLM.

### What the system returns

For each candidate, the system generates information such as:

* Candidate name
* Match score
* Matching skills
* Missing skills
* Candidate summary
* Strengths
* Weaknesses
* Hiring recommendation
* Reason behind the recommendation

The output is validated using Pydantic and returned in a consistent JSON structure.

### Technologies Used

* Python
* LangChain
* FAISS
* Pydantic
* OpenRouter
* Scikit-learn
* Pandas
* NumPy
* PyPDFLoader
* Joblib

### Embeddings

For the retrieval part, I used a local `HashingVectorizer`-based embedding approach rather than calling an external embedding API.

The implementation creates normalized 512-dimensional vectors that can be indexed and searched using FAISS.

### Testing

I tested the system with three different resumes against the same Senior Data Scientist job description:

* Alex Chen — Data Scientist
* Priya Sharma — Machine Learning Engineer
* Rahul Verma — Frontend Developer

The tests were useful for checking whether the system could identify relevant skills as well as gaps between a candidate's resume and the job requirements.

The resulting evaluations produced different match scores and recommendations based on the documented skills and experience in each resume.

---

# 2. Insurance Claim Processing Agent

The second project is a multi-agent insurance claim processing workflow built with LangGraph.

The idea is to break the claim processing task into smaller jobs instead of asking one AI model to handle everything.

Different agents check different parts of a claim, and their results are brought together before deciding what should happen next.

---

## How the Workflow Works

```text
                    START
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
   Document       Eligibility     Fraud
   Verification      Check       Detection
        |             |             |
        +-------------+-------------+
                      |
                      v
                Claim Summary
                      |
             +--------+--------+
             |        |        |
             v        v        v
        Auto Approve Reject  Human Review
                                |
                                v
                               END
```

The first three checks run independently:

1. Document verification
2. Policy eligibility
3. Fraud detection

Their results are then combined by the claim summary agent, which decides whether the claim can be approved, rejected, or needs human review.

---

## Agents Used

### Document Verification Agent

Checks whether the required documents have been submitted.

The current workflow checks for:

* Government ID
* Medical Invoice
* Official Incident Report

If mandatory documents are missing, the claim can be rejected.

### Eligibility Check Agent

This agent checks whether the policy is active and whether the incident happened within the policy coverage period.

It uses the policy status, policy expiry date, and incident date to determine eligibility.

### Fraud Detection Agent

The fraud detection part uses the OpenRouter LLM to look at the claim amount, claim type, timing, and claim notes.

It produces:

* Fraud risk level
* Risk score
* Suspicious indicators
* Reasoning

There is also a rule-based fallback so the workflow can still produce a result if the LLM response cannot be processed.

### Claim Summary Agent

This agent brings the results together and decides the next step.

The current workflow can:

* Approve a normal claim automatically
* Reject a claim with failed documentation or eligibility
* Send higher-risk or higher-value claims for human review

Claims above $10,000 are routed for human review, while medium or high fraud risk can also trigger escalation.

### Human Approval Agent

Claims that require additional review are passed to a simulated claims specialist.

The agent considers the fraud risk and identified indicators before producing the final human-review status.

---

# LangGraph Workflow

The complete workflow is built using `StateGraph`.

The graph contains:

* Parallel execution from the starting point
* Multiple verification agents
* Result aggregation
* Conditional routing
* Human review
* Final workflow termination

This was one of the main parts of the project because it demonstrates how multiple AI/logic components can work together instead of relying on a single LLM call.

---

# Claim Scenarios Tested

I tested the workflow using different types of insurance claims.

| Scenario | Situation                           | Workflow Result       |
| -------- | ----------------------------------- | --------------------- |
| 1        | Complete claim with valid documents | Auto approval         |
| 2        | Required documents missing          | Rejected              |
| 3        | Policy expired                      | Rejected              |
| 4        | High-value claim                    | Sent for human review |
| 5        | Suspicious claim                    | Sent for human review |

These scenarios were created to check different branches of the LangGraph workflow rather than testing only a normal claim.

For example, the first scenario passed the document and eligibility checks and was automatically approved.

The second scenario was rejected because required documents were missing.

The third scenario was rejected because the policy was inactive and the incident occurred after the policy expiry date.

---

# Technologies Used

### Programming

* Python
* Pandas
* NumPy

### Generative AI

* OpenRouter
* OpenRouter Free Model
* LangChain

### Agentic AI

* LangGraph
* StateGraph
* Conditional Routing
* Multi-Agent Workflow
* Human-in-the-Loop

### Resume Project

* FAISS
* Pydantic
* PyPDFLoader
* HashingVectorizer
* Joblib

### Machine Learning

* Scikit-learn
* Random Forest
* StandardScaler

---

# What I Learned

Working on these projects gave me practical experience with several concepts that are important in modern AI development.

Some of the main areas covered were:

* Building a RAG pipeline
* Working with PDF documents
* Vector similarity search
* FAISS
* Prompt design
* Structured LLM responses
* Pydantic validation
* LangChain pipelines
* Building multi-agent workflows
* Parallel agent execution
* Conditional routing
* Human-in-the-loop processing
* Connecting LLMs with traditional Python logic
* Saving and loading ML models with Joblib

---

# Future Improvements

## Resume Screening Assistant

Some improvements I would like to make in a future version include:

* Support for DOCX resumes
* Better semantic embedding models
* Processing multiple resumes at once
* Resume ranking dashboard
* ATS integration
* More detailed recruiter analytics
* Better evaluation of experience and seniority

## Insurance Claim Processing Agent

The insurance workflow could be extended with:

* OCR for uploaded documents
* Real policy database integration
* More detailed fraud detection
* Claim history analysis
* A proper adjuster dashboard
* Persistent claim records
* Audit logs
* Authentication and access control
* Integration with external insurance systems

---

# Project Summary

These two projects explore different ways of using modern AI systems.

The **Resume Screening Assistant** focuses on retrieval and document-based LLM processing, while the **Insurance Claim Processing Agent** focuses on coordinating multiple agents through a structured workflow.

Together, they provided practical experience in building AI applications using RAG, vector search, LLMs, LangChain, and LangGraph.
