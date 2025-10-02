"""
LLM client for interfacing with different language models.
"""

import openai
import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model: str = "gpt-3.5-turbo-0125"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with retry logic."""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        # Set API key from config or environment
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from OpenAI model."""
        try:
            # Merge kwargs with default config
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            }
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self.generate(messages, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")
                    raise last_exception


class HuggingFaceClient(BaseLLMClient):
    """Client for Hugging Face models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("transformers and torch are required for HuggingFace models")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from Hugging Face model."""
        try:
            import torch
            
            # Format messages into a single prompt
            prompt = self._format_messages(messages)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self.generate(messages, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")
                    raise last_exception
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai is required for Gemini models")
        
        # Set API key from config or environment
        api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(config.model)
        
        # Set up generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens
        )
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from Gemini model."""
        try:
            # Format messages into a single prompt for Gemini
            prompt = self._format_messages(messages)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            if response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini model")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            raise
    
    def generate_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self.generate(messages, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"All {self.config.max_retries} attempts failed")
                    raise last_exception
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single prompt for Gemini."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System Instructions: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)


class LLMClient:
    """Main LLM client that handles different model types."""
    
    def __init__(self, 
                 model: str = "gpt-3.5-turbo-0125",
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize LLM client.
        
        Args:
            model: Model name or path
            api_key: API key for the model
            **kwargs: Additional configuration parameters
        """
        self.config = LLMConfig(
            model=model,
            api_key=api_key,
            **kwargs
        )
        
        # Determine client type based on model
        if self._is_gemini_model(model):
            self.client = GeminiClient(self.config)
        elif self._is_openai_model(model):
            self.client = OpenAIClient(self.config)
        elif self._is_huggingface_model(model):
            self.client = HuggingFaceClient(self.config)
        else:
            # Default to OpenAI-compatible client
            self.client = OpenAIClient(self.config)
        
        logger.info(f"Initialized LLM client for model: {model}")
    
    def _is_gemini_model(self, model: str) -> bool:
        """Check if model is a Gemini model."""
        gemini_models = ["gemini", "gemini-pro", "gemini-flash", "gemini-2.0"]
        return any(gemini_model in model.lower() for gemini_model in gemini_models)
    
    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        openai_models = ["gpt-3.5-turbo", "gpt-4", "text-davinci", "code-davinci"]
        return any(openai_model in model for openai_model in openai_models)
    
    def _is_huggingface_model(self, model: str) -> bool:
        """Check if model is a Hugging Face model."""
        hf_patterns = ["mistral", "llama", "falcon", "vicuna", "/"]
        return any(pattern in model.lower() for pattern in hf_patterns)
    
    def generate(self, 
                 prompt: Union[str, List[Dict[str, str]]], 
                 system_prompt: Optional[str] = None,
                 **kwargs) -> str:
        """
        Generate response from the LLM.
        
        Args:
            prompt: User prompt or list of messages
            system_prompt: System prompt to prepend
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Convert prompt to messages format
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt
        
        return self.client.generate_with_retry(messages, **kwargs)
    
    def batch_generate(self, 
                      prompts: List[Union[str, List[Dict[str, str]]]], 
                      system_prompt: Optional[str] = None,
                      **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt to prepend
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for prompt in prompts:
            try:
                response = self.generate(prompt, system_prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing prompt: {e}")
                responses.append("")  # Return empty string on error
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "client_type": type(self.client).__name__
        }