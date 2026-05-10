package com.meta.safepill.inference;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
public class HttpClientConfig {
    @Bean
    RestClient.Builder restClientBuilder() {
        return RestClient.builder();
    }
}
