CREATE TABLE IF NOT EXISTS pills (
    id BIGSERIAL PRIMARY KEY,
    code VARCHAR(64) NOT NULL UNIQUE,
    pill_name VARCHAR(255) NOT NULL,
    manufacturer VARCHAR(255),
    dosage_form VARCHAR(100),
    shape VARCHAR(100),
    color VARCHAR(100),
    imprint_text VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingredients (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS pill_ingredients (
    pill_id BIGINT NOT NULL REFERENCES pills(id) ON DELETE CASCADE,
    ingredient_id BIGINT NOT NULL REFERENCES ingredients(id) ON DELETE RESTRICT,
    strength_text VARCHAR(100),
    PRIMARY KEY (pill_id, ingredient_id)
);

CREATE TABLE IF NOT EXISTS warnings (
    id BIGSERIAL PRIMARY KEY,
    pill_id BIGINT NOT NULL REFERENCES pills(id) ON DELETE CASCADE,
    warning_type VARCHAR(100) NOT NULL,
    warning_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interactions (
    id BIGSERIAL PRIMARY KEY,
    pill_id BIGINT NOT NULL REFERENCES pills(id) ON DELETE CASCADE,
    target_ingredient VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    interaction_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pills_name ON pills (pill_name);
CREATE INDEX IF NOT EXISTS idx_pills_imprint ON pills (imprint_text);
CREATE INDEX IF NOT EXISTS idx_warnings_pill_id ON warnings (pill_id);
CREATE INDEX IF NOT EXISTS idx_interactions_pill_id ON interactions (pill_id);
