
CREATE DATABASE IF NOT EXISTS predictive_maintenance;

USE predictive_maintenance;

-- 1. Application machines
CREATE TABLE IF NOT EXISTS machines (
    machine_id VARCHAR(50) PRIMARY KEY,
    machine_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Original dataset records
CREATE TABLE IF NOT EXISTS dataset_records (
    dataset_record_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    udi BIGINT NOT NULL UNIQUE,
    product_id VARCHAR(50),
    product_type CHAR(1),
    raw_features JSON,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Actual ML prediction history
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    machine_id VARCHAR(50) NOT NULL,
    dataset_record_id BIGINT NULL,

    input_features JSON NOT NULL,

    predicted_failure TINYINT NOT NULL,
    failure_probability DECIMAL(8,7) NOT NULL,
    threshold DECIMAL(8,7) NOT NULL,

    status VARCHAR(30) NOT NULL,
    model_name VARCHAR(100) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_prediction_machine
        FOREIGN KEY (machine_id)
        REFERENCES machines(machine_id),

    CONSTRAINT fk_prediction_dataset
        FOREIGN KEY (dataset_record_id)
        REFERENCES dataset_records(dataset_record_id)
        ON DELETE SET NULL,

    CONSTRAINT chk_prediction
        CHECK (predicted_failure IN (0, 1)),

    CONSTRAINT chk_probability
        CHECK (failure_probability BETWEEN 0 AND 1),

    CONSTRAINT chk_threshold
        CHECK (threshold BETWEEN 0 AND 1),

    INDEX idx_machine_id (machine_id),
    INDEX idx_dataset_record (dataset_record_id),
    INDEX idx_prediction_date (created_at)
);
