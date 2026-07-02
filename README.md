PMSSS Scholarship Management System
Overview

The PMSSS Scholarship Management System is a web-based application developed to automate scholarship application processing and eligibility verification. The system enables students to submit applications along with supporting documents and uses Computer Vision (CV) and OCR technologies to automatically validate applicant information.

The platform reduces manual verification effort by extracting and validating data from uploaded Aadhaar cards, marksheets, and income certificates before forwarding applications for administrative review.

Features
Student Portal
Submit scholarship applications online.
Upload Aadhaar card, marksheet, and income certificate.
Automatic eligibility verification.
Track application status using a unique Application ID.
Raise support tickets for rejected applications.
Automated Document Verification
Document detection using YOLOv8.
OCR-based text extraction using:
Tesseract OCR
EasyOCR
Aadhaar number verification.
Marks percentage extraction and validation.
Annual income extraction and validation.
Automatic eligibility decision generation.
Admin Portal
View all scholarship applications.
Review applications flagged for manual verification.
Approve or reject applications.
Add review notes.
Access application statistics and reports.
Eligibility Rules

The system automatically checks:

Aadhaar number in the uploaded document matches the entered Aadhaar number.
Marks percentage is greater than 70%.
Annual family income is below ₹8,00,000.

Applications failing any condition are automatically rejected and provided with rejection reasons.

Technology Stack
Frontend
HTML
JavaScript
CSS
Backend
FastAPI
Python
Database
MySQL
Computer Vision & OCR
YOLOv8
Tesseract OCR
EasyOCR
OpenCV
Project Structure
pmsss-project/
│
├── frontend/
│   ├── student.html
│   └── admin.html
│
├── backend/
│   ├── main.py
│   ├── schema.sql
│   ├── requirements.txt
│   └── yolov8n.pt
│
└── .gitignore
