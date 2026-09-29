"""Rank fusion and optional passage reranking without storage concerns."""
from . import tracing

import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple


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
    elif reranker_setting == "openrouter":
        from .openrouter_pipeline import OpenRouterReranker
        self.reranker_type = "openrouter"
        self.reranker = OpenRouterReranker(self.config)
        print(f"[RAG V2] Using OpenRouter reranker: {self.reranker.model_name}")
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


def fuse_rankings(self, vector_results: List[Tuple], bm25_results: List[Tuple], k: int = 60,
                  on_ranked: Callable[[List[Tuple[Dict, float]]], None] | None = None) -> List[Dict]:
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
    if on_ranked is not None:
        on_ranked(rrf_results)
    return [doc for doc, score in rrf_results]

def rerank_documents(self, docs: List[Dict], query: str, k: int) -> List[Dict]:
    """Rerank bounded passage text, retaining input order if inference fails."""
    if not docs:
        return docs
    if len(docs) < 2 and self.reranker_type != "openrouter":
        tracing.annotate(skipped_reason='too_few_candidates', score_kind='retained_source')
        tracing.passages('selected', docs)
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

            # Bound per-forward memory without dropping candidates or serializing passages.
            for start in range(0, len(docs), 8):
                input_ids_list = []
                for doc in docs[start:start + 8]:
                    text = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc['content']}"
                    inputs = self.reranker_tokenizer(
                        text, return_attention_mask=False, add_special_tokens=False,
                        truncation=True, max_length=8192 - len(prefix_tokens) - len(suffix_tokens),
                    )
                    input_ids_list.append(prefix_tokens + inputs['input_ids'] + suffix_tokens)

                max_len = max(map(len, input_ids_list))
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
                    batch_scores = outputs.logits[:, -1, [token_false_id, token_true_id]]
                    scores.extend(batch_scores.log_softmax(dim=1)[:, 1].exp().tolist())

                del inputs_tensor, attention_mask_tensor, outputs, batch_scores
            torch.cuda.empty_cache()
        elif hasattr(self, 'reranker') and self.reranker:
            # Let the model tokenizer enforce its token budget, not a character cutoff.
            pairs = []
            for doc in docs:
                source = Path(doc.get('file_path') or '').name
                passage = f"Source: {source}\n{doc['content']}" if source else doc['content']
                pairs.append([query, passage])
            scores = self.reranker.predict(pairs)
        else:
            scores = [1.0] * len(docs)

        if len(scores) != len(docs) or not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker must return one finite score per passage")
        scores = [float(score) for score in scores]

        # Sort documents by score (higher is better)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        tracing.passages('ranked', [dict(doc, score=score) for doc, score in ranked])
        tracing.annotate(score_kind='reranker')
        tracing.passages('selected', [dict(doc, score=score) for doc, score in ranked[:k]])

        return [doc for doc, score in ranked[:k]]

    except Exception as e:
        if self.reranker_type == "openrouter":
            raise
        print(f"[RAG Core] CrossEncoder reranking failed: {type(e).__name__}, falling back to original order")
        tracing.annotate(skipped_reason='reranker_failed', score_kind='retained_source')
        tracing.passages('selected', docs[:k])
        return docs[:k]
