"""
Multi-dimensional reward function for evaluating LLM outputs.
Based on paper configuration with 4 components:
1. Format: Structural correctness
2. Correctness: Answer accuracy
3. Cleanliness: Output conciseness
4. Consistency: Reasoning coherence
"""
import re
from typing import List, Dict, Union, Optional

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    print("Warning: 'sentence-transformers' library not found. Semantic matching will be disabled.")
    SentenceTransformer, cos_sim = None, None

try:
    if SentenceTransformer:
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully from cache")
    else:
        embedder = None
except Exception as e:
    print(f"Warning: Failed to load SentenceTransformer model. Error: {e}")
    embedder = None

WEIGHTS = {
    "format": 0.05,
    "correctness": 0.7,
    "cleanliness": 0.15,
    "consistency": 0.1,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Sum of WEIGHTS must be 1.0"


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', re.sub(r'[.,!?"\'`]', '', text.lower())).strip()


def _compute_correctness_reward(extracted_pred: str, ground_truth: str, semantic_threshold: float = 0.85):
    if not extracted_pred:
        return {"score": 0.0, "reason": "Correctness: No content inside box"}

    norm_pred, norm_gt = normalize(extracted_pred), normalize(ground_truth)
    if norm_pred == norm_gt:
        return {"score": 1.0, "reason": "Correctness: Exact Match"}

    pred_words, gt_words = set(norm_pred.split()), set(norm_gt.split())
    if gt_words and gt_words.issubset(pred_words) and not {'no', 'not'}.intersection(pred_words - gt_words):
        return {"score": 0.75, "reason": "Correctness: Subset Word Match"}

    if embedder:
        try:
            sim = cos_sim(embedder.encode(extracted_pred), embedder.encode(ground_truth)).item()
            if sim >= semantic_threshold:
                score = round(0.4 + (sim - semantic_threshold) / (1.0 - semantic_threshold) * 0.2, 2)
                return {"score": score, "reason": f"Correctness: Semantic Match (Sim: {sim:.2f})"}
        except Exception:
            pass

    return {"score": 0.0, "reason": "Correctness: No Match"}


def _compute_format_reward(pred: str):
    score, reasons = 1.0, []

    if pred.count('<answer>') != 1 or pred.count('</answer>') != 1:
        score -= 0.5
        reasons.append("Format: Multiple/Missing answer tags")

    if pred.count('\\boxed{') != 1:
        score -= 0.5
        reasons.append("Format: Multiple/Missing boxed")
    elif not re.search(r'<answer>.*\\boxed\{.*\}.*</answer>', pred, re.DOTALL):
        score -= 0.5
        reasons.append("Format: Boxed outside answer tag")

    if '\\\\boxed' in pred:
        score -= 0.1
        reasons.append("Format: Used double backslash")

    return {"score": max(0.0, score), "reason": ", ".join(reasons) if reasons else "Format: OK"}


def _compute_cleanliness_reward(pred: str):
    score, reasons = 1.0, []
    rep_count = pred.count('<answer>')

    if rep_count > 1:
        score -= min(0.8, 0.2 * (rep_count - 1))
        reasons.append(f"Cleanliness: Pathological Repetition ({rep_count} times)")

    if rep_count == 1:
        match = re.search(r'</answer>(.*)', pred, re.DOTALL)
        if match and match.group(1) and not match.group(1).isspace():
            score -= 0.6
            reasons.append("Cleanliness: Tailing content")

    if len(pred) > 2000:
        score -= 0.4
        reasons.append("Cleanliness: Overall length excessive")

    return {"score": max(0.0, score), "reason": ", ".join(reasons) if reasons else "Cleanliness: OK"}


def _compute_consistency_reward(pred: str, edit_context: Optional[Dict[str, str]] = None) -> Dict[str, Union[float, str]]:
    if not edit_context or 'original_knowledge' not in edit_context:
        return {"score": 1.0, "reason": "Consistency: Skipped (no context)"}

    original_knowledge = edit_context['original_knowledge']

    match = re.search(r"Re-evaluate Knowledge Network.*?(Synthesize|Formulate|<answer>)", pred, re.IGNORECASE | re.DOTALL)
    if not match:
        return {"score": 0.0, "reason": "Consistency: Re-evaluation section not found"}

    re_eval_text = match.group(0)
    norm_orig_knowledge = normalize(original_knowledge)
    norm_re_eval_text = normalize(re_eval_text)

    if norm_orig_knowledge in norm_re_eval_text:
        positive_markers = ['correct', 'valid', 'remains']
        negative_markers = ['incorrect', 'invalid', 'superseded', 'no longer', 'outdated']

        pattern = r'(\b\w+\b\s*){0,15}' + re.escape(norm_orig_knowledge) + r'(\s*\b\w+\b){0,15}'
        context_match = re.search(pattern, norm_re_eval_text)

        if context_match:
            context_text = context_match.group(0)
            if any(marker in context_text for marker in negative_markers):
                return {"score": 1.0, "reason": "Consistency: Correctly invalidated old knowledge"}
            elif any(marker in context_text for marker in positive_markers):
                return {"score": 0.0, "reason": "Consistency: Incorrectly validated old knowledge"}

    return {"score": 0.5, "reason": "Consistency: Old knowledge not explicitly invalidated"}


def _compute_full_reward_for_single_pred(
    pred: str,
    ground_truth: str,
    semantic_threshold: float,
    edit_context: Optional[Dict[str, str]] = None
) -> Dict[str, Union[float, str, Dict]]:
    if not isinstance(pred, str) or not pred.strip():
        return {"overall": 0.0, "details": {}, "reason": "Fatal: Prediction is not a valid string."}

    r_format_info = _compute_format_reward(pred)
    r_cleanliness_info = _compute_cleanliness_reward(pred)
    r_consistency_info = _compute_consistency_reward(pred, edit_context)

    extracted_pred = ""
    if r_format_info["score"] > 0:
        match = re.search(r'\\boxed\{([^}]+)\}', pred)
        if match:
            extracted_pred = match.group(1).strip()

    r_correctness_info = _compute_correctness_reward(extracted_pred, ground_truth, semantic_threshold)

    w = WEIGHTS
    if r_format_info["score"] == 0:
        overall_score = 0.0
        final_reason = r_format_info["reason"]
    else:
        overall_score = (w['format'] * r_format_info['score'] +
                         w['correctness'] * r_correctness_info['score'] +
                         w['cleanliness'] * r_cleanliness_info['score'] +
                         w['consistency'] * r_consistency_info['score'])
        final_reason = "; ".join([
            r_format_info['reason'],
            r_correctness_info['reason'],
            r_cleanliness_info['reason'],
            r_consistency_info['reason']
        ])

    details = {
        "r_format": r_format_info["score"],
        "r_correctness": r_correctness_info["score"],
        "r_cleanliness": r_cleanliness_info["score"],
        "r_consistency": r_consistency_info["score"],
    }

    return {"overall": round(overall_score, 4), "details": details, "reason": final_reason}


def reward_function_v2(
    pred: Union[str, List[str]],
    ground_truth: Union[str, List[str]],
    semantic_threshold: float = 0.85,
    edit_context: Optional[Union[Dict, List[Dict]]] = None
) -> List[Dict[str, float]]:
    """Main reward function (V2.2 - paper configuration)."""
    is_batch = isinstance(pred, list)
    if not is_batch:
        pred, ground_truth = [pred], [ground_truth]
        if edit_context:
            edit_context = [edit_context]

    all_results = []
    print("\n" + "="*20 + " [Reward Function V2.2] Starting Batch Processing " + "="*18)

    for i, (p, gt) in enumerate(zip(pred, ground_truth)):
        ctx = edit_context[i] if edit_context and i < len(edit_context) else None
        single_result = _compute_full_reward_for_single_pred(p, gt, semantic_threshold, ctx)
        all_results.append(single_result)

        print("-" * 70)
        print(f"  [Sample #{i+1}]")
        print(f"  - Prediction: {p[:200]}...")
        print(f"  - Ground Truth: {gt}")
        print(f"  - Context: {ctx}")
        print(f"  - Final Score: {single_result['overall']:.4f}")

        details = single_result['details']
        print(f"  - Breakdown: Fmt={details.get('r_format', 0):.2f}, "
              f"Corr={details.get('r_correctness', 0):.2f}, "
              f"Clean={details.get('r_cleanliness', 0):.2f}, "
              f"Consist={details.get('r_consistency', 0):.2f}")
        print(f"  - Reason: {single_result['reason']}")

    print("-" * 70)
    print(f"===== Batch Processing Finished. Total: {len(all_results)} items. " + "="*20 + "\n")

    final_results = [{
        "overall": res["overall"],
        "r_format": res["details"]["r_format"],
        "r_correctness": res["details"]["r_correctness"],
        "r_cleanliness": res["details"]["r_cleanliness"],
        "r_consistency": res["details"]["r_consistency"]
    } for res in all_results]

    return final_results


def designed_reward(
    pred: Union[str, List[str]],
    ground_truth: Union[str, List[str]],
    **kwargs
) -> List[Dict[str, float]]:
    """Compatibility alias for reward_function_v2."""
    return reward_function_v2(pred, ground_truth, **kwargs)
