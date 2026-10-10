"""Shared public workflow defaults; model revisions live in models.lock.json.

Changing decode/renderer settings changes the existing stage fingerprints.
QA thresholds remain in config/asr-quality.json, ML contracts in defaults.json.
"""

DEFAULT_PROFILE = "full"
DEFAULT_REVIEW_POLICY = "auto-prefer-retry"
REVIEW_POLICIES = ("auto-prefer-retry", "conservative", "human")
RETRY_BUDGET_SECONDS = 900.0
TURN_GROUP_GAP_SECONDS = 0.75
OFFLINE = True
ASR_DECODE = {
    "language": "fa",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 8,
    "num_workers": 1,
    "beam_size": 5,
    "best_of": 5,
    "word_timestamps": True,
    "vad_filter": False,
    "condition_on_previous_text": True,
    "initial_prompt": None,
    "hotwords": None,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "retry_policy_version": 1,
}
