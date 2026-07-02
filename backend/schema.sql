-- PMSSS Database Schema
-- Run: mysql -u root -p < schema.sql

CREATE DATABASE IF NOT EXISTS pmsss_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE pmsss_db;

CREATE TABLE IF NOT EXISTS applications (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    application_id      VARCHAR(25) UNIQUE NOT NULL,
    name                VARCHAR(100) NOT NULL,
    education_level     VARCHAR(50),
    aadhar_no           VARCHAR(20),
    state               VARCHAR(60),
    pin_code            VARCHAR(10),
    father_name         VARCHAR(100),
    email               VARCHAR(120),
    phone               VARCHAR(15),
    submitted_at        DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- CV Results
    cv_aadhar_match     TINYINT(1) DEFAULT NULL,
    cv_marks            FLOAT DEFAULT NULL,
    cv_income           FLOAT DEFAULT NULL,
    cv_processed        TINYINT(1) DEFAULT 0,

    -- Eligibility
    eligibility_status  ENUM('pending','approved','rejected') DEFAULT 'pending',
    rejection_reasons   TEXT DEFAULT NULL,

    -- Ticket / Admin
    ticket_raised       TINYINT(1) DEFAULT 0,
    admin_status        ENUM('pending','pending_review','approved','rejected') DEFAULT 'pending',
    admin_notes         TEXT DEFAULT NULL,
    admin_reviewed_at   DATETIME DEFAULT NULL,

    INDEX idx_app_id (application_id),
    INDEX idx_email  (email),
    INDEX idx_status (eligibility_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
