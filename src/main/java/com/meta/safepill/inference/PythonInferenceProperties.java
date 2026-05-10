package com.meta.safepill.inference;

import java.net.URI;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "safepill.python")
public record PythonInferenceProperties(URI baseUrl) {
}
