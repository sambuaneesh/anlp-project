import os
import re
import json
import time
import google.generativeai as genai
from google.generativeai.generative_models import ChatSession
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Any
load_dotenv()

TEST_DATASET = [
    {
        "id": "Paper_Example_Refutes",
        "claim": "The founder of the school was the daughter of Christopher, and his sister Kathleen was a former principal of the school.",
        "evidence": """
        St Hugh's College is a constituent college of University of Oxford. It was founded in 1886 by Elizabeth Wordsworth.
        Elizabeth Wordsworth was the daughter of Christopher Wordsworth.
        Kathleen, Christopher's sister, was Principal of Somerville College.
        """,
        "expected_label": "REFUTES"
    },
    {
        "id": "Simple_Supports",
        "claim": "The film 'Inception', directed by Christopher Nolan, was released in 2010.",
        "evidence": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
        "expected_label": "SUPPORTS"
    },
    {
        "id": "Complex_Refutes_With_Distractor",
        "claim": "The character played by Harrison Ford in 'Blade Runner' flies a Spinner, which is the same vehicle type used by the main character in 'The Fifth Element'.",
        "evidence": """
        In the 1982 film 'Blade Runner', Harrison Ford's character Rick Deckard pilots a flying car known as a Spinner.
        The main character of 'The Fifth Element', Korben Dallas, drives a flying taxi cab, which is a different model.
        The Spinner was designed by Syd Mead.
        """,
        "expected_label": "REFUTES"
    },
    {
        "id": "Multi_Hop_Supports",
        "claim": "The lead singer of the band that wrote 'Stairway to Heaven' was born in West Bromwich.",
        "evidence": """
        'Stairway to Heaven' is a song by the English rock band Led Zeppelin.
        The band's lead singer was Robert Plant. Robert Plant was born in West Bromwich, Staffordshire, England.
        """,
        "expected_label": "SUPPORTS"
    }
]

class LLMClient:
    """A client to interact with the Google Gemini API, now with chat support."""
    def __init__(self, model: str = "gemini-2.0-flash"):
        api_key = "api key here"  
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.generation_config = genai.GenerationConfig(temperature=0.0)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def query(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config, safety_settings=self.safety_settings)
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return ""

    def start_chat_session(self) -> ChatSession:
        return self.model.start_chat(history=[])

PROMPTS = {
    "claim_graph_construction": 'You are an expert in knowledge graph construction... Claim: "{claim}" Output JSON:',
    "evidence_graph_construction": 'You are an expert in knowledge graph construction... Evidence: "{evidence}" Known Entities: {known_entities} Output JSON:',
    "graph_match": 'You are a fact-checking agent... Evidence Graph: {evidence_graph_str} Claim Triplet: {triplet_str} Is the claim triplet true? Answer with only "true" or "false".',
    "graph_completion": 'You are a fact-checking agent... Evidence Graph: {evidence_graph_str} Claim Triplet: {triplet_str} What is the correct entity? Output only the entity name or "none".',
} # Using shortened prompts for brevity, but they are the same as before

class BaseGraphFC:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def _parse_json_output(self, llm_output: str) -> List[Dict[str, str]]:
        try:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_output)
            json_str = match.group(1) if match else llm_output
            parsed = json.loads(json_str)
            return [parsed] if isinstance(parsed, dict) else parsed
        except (json.JSONDecodeError, AttributeError):
            print(f"Warning: Failed to parse LLM JSON output: {llm_output}")
            return []

    def construct_claim_graph(self, claim: str):
        prompt = PROMPTS["claim_graph_construction"].format(claim=claim)
        return self._parse_json_output(self.llm.query(prompt))

    def construct_evidence_graph(self, evidence: str, claim_graph: List[Dict[str, Any]]):
        # --- THIS IS THE FIX ---
        raw_entities = [v for t in claim_graph for v in t.values()]
        
        flat_entities = []
        for entity in raw_entities:
            if isinstance(entity, list):
                flat_entities.extend(entity)
            else:
                flat_entities.append(entity)
        
            known_entities = list(set(
            e for e in flat_entities if isinstance(e, (str, int, float)) and not str(e).startswith('x')
        ))
        # --- END OF FIX ---

        prompt = PROMPTS["evidence_graph_construction"].format(evidence=evidence, known_entities=known_entities)
        return self._parse_json_output(self.llm.query(prompt))

    def plan_verification_order(self, claim_graph: List[Dict]):
        return sorted(claim_graph, key=lambda t: sum(1 for v in t.values() if str(v).startswith('x')))

    def _update_graph(self, graph: List[Dict], placeholder: str, grounded_entity: str):
        updated_graph = []
        for triplet in graph:
            updated_triplet = {k: grounded_entity if v == placeholder else v for k, v in triplet.items()}
            updated_graph.append(updated_triplet)
        return updated_graph

# --- Approach 1: Single-Turn (Stateless) ---
class SingleTurnGraphFC(BaseGraphFC):
    def _graph_match(self, triplet: Dict, evidence_graph: List[Dict]) -> bool:
        prompt = PROMPTS["graph_match"].format(
            evidence_graph_str=json.dumps(evidence_graph, indent=2),
            triplet_str=json.dumps(triplet)
        )
        return "true" in self.llm.query(prompt).lower()

    def _graph_completion(self, triplet: Dict, evidence_graph: List[Dict]) -> Optional[Tuple[str, str]]:
        prompt = PROMPTS["graph_completion"].format(
            evidence_graph_str=json.dumps(evidence_graph, indent=2),
            triplet_str=json.dumps(triplet)
        )
        response = self.llm.query(prompt)
        if response.lower() == "none" or not response: return None
        placeholder = next((v for v in triplet.values() if str(v).startswith('x')), None)
        return (placeholder, response) if placeholder else None

    def verify(self, claim: str, evidence: str):
        claim_graph = self.construct_claim_graph(claim)
        evidence_graph = self.construct_evidence_graph(evidence, claim_graph)
        verification_plan = self.plan_verification_order(claim_graph)

        for i, triplet in enumerate(verification_plan):
            unknowns = [v for v in triplet.values() if str(v).startswith('x')]
            if len(unknowns) == 0:
                if not self._graph_match(triplet, evidence_graph): return False, f"Failed match: {triplet}"
            elif len(unknowns) == 1:
                result = self._graph_completion(triplet, evidence_graph)
                if not result: return False, f"Failed completion: {triplet}"
                placeholder, grounded_entity = result
                remaining_plan = verification_plan[i+1:]
                verification_plan = verification_plan[:i+1] + self._update_graph(remaining_plan, placeholder, grounded_entity)
        return True, "All triplets verified."

# --- Approach 2: Multi-Turn (Dialog) ---
class MultiTurnDialogFC(BaseGraphFC):
    def verify(self, claim: str, evidence: str):
        claim_graph = self.construct_claim_graph(claim)
        evidence_graph = self.construct_evidence_graph(evidence, claim_graph)
        verification_plan = self.plan_verification_order(claim_graph)

        chat = self.llm.start_chat_session()
        initial_prompt = f"Let's perform a step-by-step fact check. Here is the evidence graph: {json.dumps(evidence_graph, indent=2)}. We will verify these triplets: {json.dumps(verification_plan, indent=2)}. Respond with only 'true'/'false' for matching or an entity name/'none' for completion. Acknowledge and wait for my first instruction."
        chat.send_message(initial_prompt)

        for i, triplet in enumerate(verification_plan):
            unknowns = [v for v in triplet.values() if str(v).startswith('x')]
            if len(unknowns) == 0:
                prompt = f'Is the triplet {json.dumps(triplet)} supported by the evidence?'
                response_text = chat.send_message(prompt).text.lower()
                if "true" not in response_text: return False, f"Failed match: {triplet}"
            elif len(unknowns) == 1:
                prompt = f'Resolve the placeholder in {json.dumps(triplet)}.'
                grounded_entity = chat.send_message(prompt).text
                if grounded_entity.lower() == "none" or not grounded_entity: return False, f"Failed completion: {triplet}"
                placeholder = unknowns[0]
                remaining_plan = verification_plan[i+1:]
                verification_plan = verification_plan[:i+1] + self._update_graph(remaining_plan, placeholder, grounded_entity)
        return True, "All triplets verified."

def run_experiment():
    llm_client = LLMClient()
    single_turn_fc = SingleTurnGraphFC(llm_client)
    multi_turn_fc = MultiTurnDialogFC(llm_client)

    approaches = {
        "Single-Turn (Stateless)": single_turn_fc,
        "Multi-Turn (Dialog)": multi_turn_fc
    }

    results = {name: {"correct": 0, "total": 0, "predictions": []} for name in approaches}

    for item in TEST_DATASET:
        print(f"\n" + "="*80)
        print(f"Testing Claim ID: {item['id']}")
        print(f"Claim: {item['claim']}")
        print(f"Expected: {item['expected_label']}")
        print("."*80)

        for name, model in approaches.items():
            print(f"\n--- Running {name} ---")
            start_time = time.time()
            is_supported, reason = model.verify(item['claim'], item['evidence'])
            duration = time.time() - start_time
            
            prediction = "SUPPORTS" if is_supported else "REFUTES"
            is_correct = (prediction == item['expected_label'])
            
            results[name]["total"] += 1
            if is_correct:
                results[name]["correct"] += 1
            
            results[name]["predictions"].append({
                "id": item['id'],
                "prediction": prediction,
                "expected": item['expected_label'],
                "correct": is_correct
            })
            
            print(f"Verdict: {prediction} ({'CORRECT' if is_correct else 'INCORRECT'})")
            print(f"Reason: {reason}")
            print(f"Time Taken: {duration:.2f}s")
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    for name, data in results.items():
        accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
        print(f"\nApproach: {name}")
        print(f"  Accuracy: {accuracy:.2f}% ({data['correct']}/{data['total']})")
        
        print("  Breakdown:")
        for pred in data['predictions']:
            status = "✅" if pred['correct'] else "❌"
            print(f"    {status} ID: {pred['id']:<30} -> Predicted: {pred['prediction']:<10}, Expected: {pred['expected']}")

if __name__ == "__main__":
    run_experiment()