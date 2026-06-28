"""
jd_config.py
------------
Single source of truth for the Job Description signal configuration.
V2 Architecture: Timeline Rules, AI Depth scoring, Seniority.
"""

# ── Core must-have skills (from JD "Things you absolutely need") ────────────
JD_MUST_HAVE_SKILLS = {
    # Retrieval & Search
    "embeddings", "embedding", "sentence-transformers", "bge", "e5",
    "vector search", "semantic search", "hybrid search", "bm25",
    "faiss", "qdrant", "weaviate", "pinecone", "milvus",
    "opensearch", "elasticsearch",
    "retrieval", "information retrieval", "rag",
    # Ranking & Recommendation
    "ranking", "reranking", "learning to rank", "ltr",
    "recommendation", "recommendation system", "recommender",
    # Evaluation
    "ndcg", "mrr", "map", "a/b testing", "ab testing",
    "eval framework", "offline evaluation", "online evaluation",
    # Core tech
    "python", "pytorch", "nlp", "transformers", "bert",
    "huggingface", "hugging face",
    # LLM (must have DEPTH not just usage)
    "fine-tuning", "fine tuning", "llm", "large language model",
}

JD_NICE_HAVE_SKILLS = {
    "lora", "qlora", "peft", "xgboost", "lightgbm",
    "airflow", "spark", "dbt", "kafka",
    "docker", "kubernetes", "redis",
    "distributed inference", "model serving", "triton",
    "open source", "github",
}

# ── Disqualify Domains (pure CV / speech / robotics) ─────────────────────────
JD_DISQUALIFY_DOMAINS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "yolo", "opencv", "resnet",
    "speech recognition", "asr", "text to speech", "tts", "speech synthesis",
    "robotics", "ros", "slam", "autonomous driving",
    "point cloud", "lidar",
}

# ── Technology Timeline (Honeypot detection rules) ───────────────────────────
TECHNOLOGY_TIMELINES = {
    "chatgpt": 2022,
    "gpt-4": 2023,
    "gpt-3": 2020,
    "llm": 2018,  # General LLM terminology timeline
    "transformers": 2017,
    "bert": 2018,
    "openai": 2016,
    "anthropic": 2021,
    "langchain": 2022,
    "llamaindex": 2022,
    "pinecone": 2021,
    "weaviate": 2019,
    "qdrant": 2021,
    "milvus": 2019,
    "tensorflow": 2015,
    "pytorch": 2016,
}

# ── Seniority Keywords ────────────────────────────────────────────────────────
SENIORITY_TITLES = {
    "staff", "principal", "architect", "lead", "head", "director", "vp"
}

SENIORITY_VERBS = {
    "architected", "led", "mentored", "directed", "managed",
    "spearheaded", "owned", "scaled"
}

# ── AI Depth Configuration (4-Level Score) ───────────────────────────────────
# AI Nouns
AI_NOUNS = {
    "rag", "llm", "recommendation", "ranking", "search", "vector",
    "embeddings", "fine-tuning", "model", "pipeline", "system"
}

# AI Depth Rubric (proximities)
AI_LEVELS = {
    "L4_LEADERSHIP": {
        "verbs": {"architected", "led", "owned", "spearheaded", "designed"},
        "score": 20
    },
    "L3_PRODUCTION": {
        "verbs": {"deployed", "scaled", "optimized", "productionized", "shipped"},
        "score": 10
    },
    "L2_IMPLEMENTATION": {
        "verbs": {"built", "developed", "implemented", "created", "engineered"},
        "score": 5
    },
    "L1_AWARENESS": {
        "verbs": {"used", "worked on", "assisted", "helped", "experimented", "researched"},
        "score": 1
    }
}

# AI Depth Penalties
AI_TUTORIAL_KEYWORDS = {"tutorial", "chatbot", "course project", "bootcamp", "toy project"}
AI_BUZZWORDS = {"langchain", "crewai", "autogen", "pinecone", "weaviate"}

# ── Preferred Locations ───────────────────────────────────────────────────────
PREFERRED_CITIES = {
    "pune", "noida", "bangalore", "bengaluru", "hyderabad",
    "mumbai", "delhi", "new delhi", "gurugram", "gurgaon",
    "delhi ncr", "ncr",
}

# ── Feature Weights (Stage 3 Re-ranking) ─────────────────────────────────────
WEIGHTS = {
    "semantic_match": 0.35,
    "ai_depth": 0.25,
    "seniority": 0.15,
    "experience_maturity": 0.10,
    "behavioral": 0.10,
    "availability": 0.05,
}

SALARY_MIN_FIT = 20
SALARY_MAX_FIT = 100
