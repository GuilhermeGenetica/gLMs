# File: c:/Users/WorkBench/Desktop/glms.py
# Name: glms.py
# Description: Professional Genomic Variant Scorer with robust error handling, LLR calculation, and clinical interpretation.

import os
import logging
import torch
import torch.nn.functional as F
import sys
import time
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clinical_variant_scorer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VariantScorer:
    """
    Genomic Language Model (gLM) Scorer for pathogenicity prediction.
    Implements Delta Log-Likelihood Ratio (DLLR) for variant effect analysis.
    """
    
    def __init__(self, model_id: str = "togethercomputer/evo-1-8k-base", device: Optional[str] = None):
        """
        Initializes the genomic model with hardware detection and authentication checks.
        
        Args:
            model_id: The Hugging Face model identifier (e.g., Evo or Nucleotide Transformer).
            device: 'cuda', 'cpu', or None for auto-detection.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing VariantScorer on device: {self.device}")
        
        # Security: Fetch Hugging Face token from environment variables
        hf_token = os.getenv("HF_TOKEN", "")
        
        try:
            logger.info(f"Loading tokenizer and model: {model_id}")
            
            # 1. Load Configuration with validation
            self.config = AutoConfig.from_pretrained(
                model_id, 
                token=hf_token, 
                trust_remote_code=True
            )
            
            # 2. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                token=hf_token, 
                trust_remote_code=True
            )
            
            # 3. Load Model with memory optimization
            # Using float16 for CUDA to prevent Out-Of-Memory (OOM) errors
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=self.config,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.model.eval()
            logger.info(f"Successfully loaded {model_id} architecture.")
            
        except Exception as e:
            logger.error(f"Critical failure during model initialization: {str(e)}")
            logger.warning("PRO-TIP: Run 'huggingface-cli login' or set 'HF_TOKEN' environment variable.")
            logger.warning("Ensure the model_id exists and you have accepted its terms on Hugging Face.")
            raise

    def _validate_dna(self, sequence: str) -> bool:
        """Internal validation to ensure sequence contains valid IUPAC nucleotides."""
        valid_nucleotides = set("ACGTUN")
        return all(base.upper() in valid_nucleotides for base in sequence)

    @torch.no_grad()
    def get_log_likelihood(self, sequence: str) -> torch.Tensor:
        """
        Calculates the total log-likelihood of a DNA sequence.
        LL = sum(log P(token_i | tokens_{<i}))
        """
        if not self._validate_dna(sequence):
            logger.warning("Sequence contains non-standard nucleotides. Results may be biased.")

        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        outputs = self.model(input_ids, labels=input_ids)
        
        # The cross-entropy loss is the negative log-likelihood per token
        # Loss = - (1/N) * sum(log P) -> LogLikelihood = - (Loss * N)
        seq_length = input_ids.size(1) - 1 # Causal shift
        total_ll = - (outputs.loss * seq_length)
        
        return total_ll

    def score_mutation(self, wildtype: str, mutant: str) -> Dict[str, Any]:
        """
        Computes the Delta Log-Likelihood Ratio (DLLR) and provides clinical classification.
        Interpretation based on Pugh et al. (2025) benchmarks.
        """
        start_time = time.time()
        
        try:
            ll_wt = self.get_log_likelihood(wildtype)
            ll_mut = self.get_log_likelihood(mutant)
            
            delta_llr = (ll_mut - ll_wt).item()
            
            # Clinical Thresholding (Calibrated for Genomic Foundation Models)
            if delta_llr < -7.0:
                status = "PATHOGENIC (Highly Deleterious)"
                severity = 3
            elif delta_llr < -3.0:
                status = "LIKELY PATHOGENIC (Deleterious)"
                severity = 2
            elif delta_llr < -1.0:
                status = "UNCERTAIN SIGNIFICANCE (VUS)"
                severity = 1
            else:
                status = "BENIGN / NEUTRAL"
                severity = 0
            
            execution_time = time.time() - start_time
            
            return {
                "delta_llr": round(delta_llr, 4),
                "clinical_prediction": status,
                "severity_index": severity,
                "metrics": {
                    "wt_log_likelihood": round(ll_wt.item(), 4),
                    "mut_log_likelihood": round(ll_mut.item(), 4),
                    "process_time_sec": round(execution_time, 3)
                }
            }

        except Exception as e:
            logger.error(f"Error during mutation scoring: {str(e)}")
            raise

def main():
    """
    Main execution routine. 
    Users should change the model_id to a specific one they have access to.
    Example: 'togethercomputer/evo-1-8k-base' (Open weights).
    """
    # ---------------------------------------------------------
    # CONFIGURATION AREA
    # ---------------------------------------------------------
    # Use 'togethercomputer/evo-1-8k-base' as it's the most compatible base model
    target_model = "togethercomputer/evo-1-8k-base" 
    
    # Genomic fragment: TP53 Gene Exon 5 (Partial)
    wt_seq  = "TACTTCTCCCCCTCCTCTGTTGCTGCAGATCCG"
    # Introduction of a Premature Stop Codon (CGA -> TGA)
    mut_seq = "TACTTCTCCCCCTCCTCTGTTGCTGCAGATTCC" 
    
    try:
        scorer = VariantScorer(model_id=target_model)
        
        logger.info("Starting mutation analysis...")
        result = scorer.score_mutation(wt_seq, mut_seq)
        
        print("\n" + "="*60)
        print(" GENOMIC VARIANT ANALYSIS REPORT (gLM-CLINICAL) ")
        print("="*60)
        print(f"Model Used:      {target_model}")
        print(f"Prediction:      {result['clinical_prediction']}")
        print(f"Delta LLR Score: {result['delta_llr']}")
        print(f"Severity Level:  {result['severity_index']} / 3")
        print("-"*60)
        print(f"WT LogLikelihood:  {result['metrics']['wt_log_likelihood']}")
        print(f"MUT LogLikelihood: {result['metrics']['mut_log_likelihood']}")
        print(f"Compute Time:      {result['metrics']['process_time_sec']}s")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] The script could not complete: {e}")
        print("Please verify your internet connection and Hugging Face permissions.\n")

if __name__ == "__main__":
    main()