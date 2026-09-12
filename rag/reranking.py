"""Rank fusion and optional passage reranking without storage concerns."""

import math
from typing import Dict, List, Tuple


def initialize_reranker(self):
    """Load the configured passage reranker, preserving optional-dependency fallbacks."""
    # Load only the configured reranker; light installs retain fusion order.
    reranker_model = self.config.get('reranker_model') or ""
    reranker_setting = str(self.config.get("reranker_type", "auto")).lower()
    self.reranker_type = "cross-encoder"
    self.reranker = None
    if reranker_setting == "auto":
        try:
            import sentence_transformers  # noqa: F401
            reranker_setting = "cross-encoder"
        except ImportError:
            reranker_setting = "disabled"
    if reranker_setting in {"disabled", "none", "off"}:
        print("[RAG V2] Reranker off.")
    elif reranker_setting == "fastembed":
        from .embeddings import FastEmbedReranker
        self.reranker = FastEmbedReranker(reranker_model or "Xenova/ms-marco-MiniLM-L-6-v2")
        print(f"[RAG V2] Using local ONNX reranker: {self.reranker.model_name}")
    else:
        reranker_model = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if "qwen3-reranker" in reranker_model.lower():
            self.reranker_type = "qwen3"
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[RAG V2] Loading Qwen3 Reranker on {self.device} from local cache: {reranker_model}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                reranker_model,
                padding_side='left',
                local_files_only=True
            )
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                reranker_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                local_files_only=True
            ).eval()

        else:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                print("[RAG V2] Reranker skipped: sentence-transformers is not installed (full extra). Results keep their fusion order.")
            else:
                self.reranker = CrossEncoder(
                    reranker_model,
                    device=self.device
                )
                print(f"[RAG V2] Using CrossEncoder reranking with model: {reranker_model}")


def fuse_rankings(self, vector_results: List[Tuple], bm25_results: List[Tuple], k: int = 60) -> List[Dict]:
    """Combine ranked evidence using weighted reciprocal rank fusion.

    Retain the first document for an ID and its last rank in each input list.
    Stable sorting preserves insertion order when fused scores tie.
    """
    doc_scores = {}
    
    weights = self.config.get("ensemble_weights", [1.0, 1.0])
    
    # Retain the first record and the last rank per source for repeated IDs.
    for rank_key, results in (("vector_rank", vector_results), ("bm25_rank", bm25_results)):
        for rank, (doc, distance) in enumerate(results, 1):
            doc_id = doc.get("id", id(doc))
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "vector_rank": float('inf'), "bm25_rank": float('inf')}
            doc_scores[doc_id][rank_key] = rank

    rrf_results = []
    for doc_id, data in doc_scores.items():
        vector_score = weights[0] * (1.0 / (k + data["vector_rank"])) if data["vector_rank"] != float('inf') else 0
        bm25_score = weights[1] * (1.0 / (k + data["bm25_rank"])) if data["bm25_rank"] != float('inf') else 0
        
        rrf_score = vector_score + bm25_score
        if rrf_score > 0:
            rrf_results.append((data["doc"], rrf_score))

    # Sort by RRF score (higher is better)
    rrf_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in rrf_results]

def legacy_llm_rerank(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
    """Preserve fused order for the unsupported legacy LLM setting."""
    print("[RAG Core] Executing LLM-based Reranking...")
    return docs[:k]


def rerank_documents(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
    """Rerank bounded passage text, retaining input order if inference fails."""
    if len(docs) <= k:
        return docs

    try:
        if self.reranker_type == "qwen3":
            import torch
            token_false_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
            token_true_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = self.reranker_tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = self.reranker_tokenizer.encode(suffix, add_special_tokens=False)
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            
            scores = []
            if not docs:
                return []

            if self.reranker_tokenizer.pad_token_id is None:
                self.reranker_tokenizer.pad_token_id = self.reranker_tokenizer.eos_token_id

            input_ids_list = []
            max_len = 0
            for doc in docs:
                text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc['content'][:1000]}"
                inputs = self.reranker_tokenizer(text, return_attention_mask=False, add_special_tokens=False)
                input_ids = prefix_tokens + inputs['input_ids'] + suffix_tokens
                input_ids_list.append(input_ids)
                if len(input_ids) > max_len:
                    max_len = len(input_ids)

            pad_token_id = self.reranker_tokenizer.pad_token_id
            padded_input_ids = []
            attention_masks = []

            for ids in input_ids_list:
                pad_len = max_len - len(ids)
                # Left padding for decoder-only models
                padded_input_ids.append([pad_token_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))

            inputs_tensor = torch.tensor(padded_input_ids).to(self.device)
            attention_mask_tensor = torch.tensor(attention_masks).to(self.device)

            with torch.no_grad():
                outputs = self.reranker_model(
                    inputs_tensor, attention_mask=attention_mask_tensor,
                    logits_to_keep=1, use_cache=False,
                )
                batch_scores = outputs.logits[:, -1, :]
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores_stack = torch.stack([false_vector, true_vector], dim=1)
                batch_scores_log = torch.nn.functional.log_softmax(batch_scores_stack, dim=1)
                scores = batch_scores_log[:, 1].exp().tolist()
                
            # Aggressively free memory
            del inputs_tensor
            del attention_mask_tensor
            del outputs
            del batch_scores
            del true_vector
            del false_vector
            del batch_scores_stack
            del batch_scores_log
            torch.cuda.empty_cache()
        elif hasattr(self, 'reranker') and self.reranker:
            # Let the model tokenizer enforce its token budget, not a character cutoff.
            pairs = [[query, doc['content']] for doc in docs]
            scores = self.reranker.predict(pairs)
        else:
            scores = [1.0] * len(docs)

        if len(scores) != len(docs) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker must return one finite score per passage")

        # Sort documents by score (higher is better)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in ranked[:k]]

    except Exception as e:
        print(f"[RAG Core] CrossEncoder reranking failed: {e}, falling back to original order")
        return docs[:k]
