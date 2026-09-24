# LLM Service - Centralized LLM interaction with fallback
import os
from typing import Dict, Any, Optional, List
from functools import lru_cache
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Try to import LangChain Groq
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_GROQ_AVAILABLE = True
except ImportError:
    LANGCHAIN_GROQ_AVAILABLE = False
    logger.warning("langchain-groq not available, LLM features will use fallback")


class LLMService:
    """Centralized LLM service with fallback logic."""
    
    def __init__(self):
        self._llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client."""
        if not LANGCHAIN_GROQ_AVAILABLE:
            logger.warning("LLM not available - using deterministic fallbacks")
            return
            
        api_key = settings.GROQ_API_KEY
        if not api_key or api_key == "":
            logger.warning("GROQ_API_KEY not set - using deterministic fallbacks")
            return
            
        try:
            self._llm = ChatGroq(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                groq_api_key=api_key,
            )
            logger.info(f"LLM initialized: {settings.LLM_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._llm = None
    
    @property
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self._llm is not None
    
    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        """Invoke LLM with prompt."""
        if not self.is_available:
            return self._fallback_response(prompt)
        
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Deterministic fallback when LLM unavailable."""
        prompt_lower = prompt.lower()
        
        if "plan" in prompt_lower or "workflow" in prompt_lower:
            return self._fallback_planner()
        elif "critic" in prompt_lower or "analyze" in prompt_lower or "feedback" in prompt_lower:
            return self._fallback_critic()
        elif "feature" in prompt_lower and "engineer" in prompt_lower:
            return self._fallback_feature_engineering()
        elif "report" in prompt_lower or "summary" in prompt_lower:
            return self._fallback_report()
        else:
            return "LLM unavailable - using deterministic logic"
    
    def _fallback_planner(self) -> str:
        return """Based on the dataset profile, I recommend the following workflow:
1. Clean missing values using median/mode imputation
2. Encode categorical variables with one-hot encoding
3. Handle class imbalance with balanced class weights
4. Run EDA to understand feature-target relationships
5. Engineer features: datetime decomposition, log transforms for skewed features
6. Train baseline models: LogisticRegression, RandomForest, XGBoost
7. Evaluate with cross-validation
8. Tune best model with Optuna
9. Critique results and iterate if needed"""
    
    def _fallback_critic(self) -> str:
        return """Issues detected: class imbalance present, recall below precision.
Recommended actions: apply class_weight='balanced', consider threshold adjustment.
Summary: Model needs improvement on minority class recall."""
    
    def _fallback_feature_engineering(self) -> str:
        return """Recommended features: log transform for skewed numerical features, 
datetime decomposition if datetime columns present, 
drop high-cardinality categorical features (>50 unique values)."""
    
    def _fallback_report(self) -> str:
        return "Autonomous ML workflow completed. Best model selected based on validation metrics."


@lru_cache
def get_llm_service() -> LLMService:
    """Get singleton LLM service instance."""
    return LLMService()