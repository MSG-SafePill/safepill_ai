package com.meta.safepill.inference;

import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class PythonInferenceClient {
    private final RestClient restClient;

    public PythonInferenceClient(RestClient.Builder restClientBuilder, PythonInferenceProperties properties) {
        this.restClient = restClientBuilder
                .baseUrl(properties.baseUrl().toString())
                .build();
    }

    public PythonServiceHealthResponse health() {
        return restClient.get()
                .uri("/health")
                .retrieve()
                .body(PythonServiceHealthResponse.class);
    }

    public PillInferenceResponse infer(PillInferenceRequest request) {
        return restClient.post()
                .uri("/infer")
                .body(request)
                .retrieve()
                .body(PillInferenceResponse.class);
    }
}
