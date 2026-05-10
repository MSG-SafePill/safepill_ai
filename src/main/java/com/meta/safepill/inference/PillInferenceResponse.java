package com.meta.safepill.inference;

import java.util.List;

public record PillInferenceResponse(String requestId, String status, List<PillCandidate> candidates) {
}
