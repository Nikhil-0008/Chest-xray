# ChestScan

## AI Powered Chest X Ray Analysis and Emergency Triage Support System

ChestScan is an AI based medical imaging support system designed to analyze chest X ray images for pneumonia detection and assist healthcare teams in prioritizing patients based on their clinical condition.

The system combines DenseNet121 based deep learning, Grad CAM explainability, computer vision validation, patient history, vital sign analysis, deterioration tracking, and an emergency triage queue into a single application.

ChestScan is designed as a clinical decision support system and not as a replacement for a qualified medical professional.

## Key Features

### Pneumonia Detection

ChestScan uses a trained DenseNet121 model to classify chest X ray images into two categories:

* NORMAL
* PNEUMONIA

The model produces a probability for each class and displays the predicted result along with its confidence.

### Chest X Ray Validation

Before performing the pneumonia prediction, the uploaded image is validated using a vision model.

The validation system checks whether the uploaded image is an actual clinical chest X ray. Images such as brain scans, other body part X rays, medical illustrations, and ordinary photographs are rejected before they reach the prediction model.

This prevents inappropriate images from being processed by the pneumonia detection model.

### Grad CAM Explainability

When pneumonia is detected, ChestScan generates a Grad CAM visualization from the final convolutional layer of DenseNet121.

The visualization highlights the region that contributed most strongly to the model prediction.

The interface provides:

* The original X ray
* The Grad CAM overlay
* The hottest activation point
* The approximate lung region
* An opacity marker showing the highlighted area

This makes the AI prediction more interpretable instead of presenting only a classification result.

### Emergency Triage Support

ChestScan includes an emergency triage system to help organize patients who are waiting for clinical attention.

Patients are categorized into four emergency levels:

* CRITICAL
* HIGH
* MODERATE
* LOW

The suggested emergency level is based on the pneumonia finding and available clinical information.

The final emergency level remains editable by clinical staff.

### Patient Priority System

When multiple patients have the same emergency level, ChestScan does not use patient name or patient ID to determine priority.

Instead, it follows a defined priority hierarchy:

1. Emergency level
2. Available clinical severity indicators
3. Deterioration trend
4. Waiting time
5. Registration timestamp as the final tie breaker

This allows two patients with the same emergency level to be differentiated using their available clinical information.

### Vital Sign Based Severity

ChestScan can use the following vital signs when they are available:

* SpO2
* Respiratory rate
* Heart rate

The system calculates a severity score from the available measurements.

Missing vital signs are not fabricated or assumed.

Normal values contribute zero to the severity score while still being recognized as available clinical information.

### Deterioration Tracking

For patients with a Patient ID, ChestScan can compare the current vital signs with the most recent previous record for the same patient.

The system can identify the patient's trend as:

* WORSENING
* STABLE
* IMPROVING
* UNKNOWN

Only vital signs available in both the current and previous records are compared.

This allows the system to consider whether the patient's condition is changing over time.

### Live Triage Queue

Saved patients who are still waiting are displayed in a live triage queue.

The queue automatically orders patients according to the priority hierarchy.

For every patient, the queue can display:

* Patient name
* Emergency level
* SpO2
* Respiratory rate
* Heart rate
* Deterioration status
* Waiting time

Healthcare staff can mark completed patients as discharged from the waiting queue.

### Patient History

ChestScan maintains a history of previously saved scans.

The history includes information such as:

* Patient details
* Prediction
* Confidence
* Emergency level
* Deterioration status
* Waiting time
* Clinical remarks
* Grad CAM preview

This provides a centralized view of previously analyzed patients.

## System Workflow

The complete ChestScan workflow is:

```text
Chest X Ray Upload
        |
        v
Chest X Ray Validation
        |
        v
Image Preprocessing
        |
        v
DenseNet121 Prediction
        |
        v
Normal or Pneumonia
        |
        +----------------------+
        |                      |
        v                      v
Normal Result            Pneumonia Result
                               |
                               v
                         Grad CAM Analysis
                               |
                               v
                     Clinical Information
                    SpO2 / RR / HR / Patient ID
                               |
                               v
                     Severity Calculation
                               |
                               v
                    Emergency Level Suggestion
                               |
                               v
                    Deterioration Detection
                               |
                               v
                     Triage Priority Calculation
                               |
                               v
                       Live Triage Queue
```

## AI Pipeline

### Image Preprocessing

The uploaded image is converted to RGB format and resized according to the model input resolution.

The image is then normalized using the preprocessing function associated with DenseNet121.

The processed image is passed to the trained model for inference.

### DenseNet121 Classification

DenseNet121 is used as the underlying convolutional neural network for chest X ray classification.

The model produces either a binary sigmoid output or class probabilities depending on the trained model architecture.

The system automatically determines the output format and converts the result into:

```text
NORMAL probability
PNEUMONIA probability
```

The class with the highest probability becomes the predicted result.

### Grad CAM

Grad CAM uses gradients from the selected convolutional layer to determine which image regions contributed to the prediction.

ChestScan uses:

```text
conv5_block16_concat
```

as the preferred Grad CAM layer.

If that layer is unavailable, the system automatically searches for a suitable four dimensional convolutional layer.

## Emergency Level Logic

The emergency level suggestion is separated from model confidence.

A high confidence pneumonia prediction does not automatically mean that the patient is in a critical condition.

When vital signs are available, they are used to estimate clinical severity.

When no vital signs are available, model confidence can provide a limited initial suggestion, but imaging confidence alone cannot produce a CRITICAL recommendation.

The suggested level can be manually changed by clinical staff.

## Patient Priority Logic

ChestScan uses the following priority order:

```text
Emergency Level
        ↓
Clinical Severity
        ↓
Deterioration Trend
        ↓
Waiting Time
        ↓
FIFO Registration Time
```

For example, if two patients are both classified as HIGH priority, their available vital signs are considered next.

If their severity is also similar, the system considers whether their condition is worsening.

If they remain equivalent, the patient who has been waiting longer receives higher queue priority.

If the waiting time is also effectively equal, the earlier registration timestamp is used.

Patient names and Patient IDs are never used to decide priority.

## Database and Storage

ChestScan uses Supabase for persistent data storage.

Patient scan records contain information including:

* Patient ID
* Patient name
* Age
* Sex
* Prediction
* Confidence
* Normal probability
* Pneumonia probability
* Grad CAM region
* SpO2
* Respiratory rate
* Heart rate
* Emergency level
* Deterioration status
* Priority score
* Triage status
* Waiting timestamp

The original X ray and Grad CAM result are stored separately using Supabase Storage.

## Technology Stack

### Frontend

Gradio

### Programming Language

Python

### Deep Learning

TensorFlow

DenseNet121

### Computer Vision

OpenCV

Pillow

Grad CAM

### AI Vision Validation

OpenRouter Vision Model

### Database

Supabase

### Data Processing

NumPy

### API and Networking

Requests

### Application Features

Patient history

Emergency triage

Priority queue

Vital sign analysis

Deterioration tracking

Clinical remarks

## Project Architecture

```text
User
 |
 v
Gradio Interface
 |
 +--------------------------+
 |                          |
 v                          v
X Ray Validation       Patient Information
 |                          |
 v                          |
DenseNet121                 |
 |                          |
 v                          |
Prediction                  |
 |                          |
 v                          |
Grad CAM                    |
 |                          |
 +------------+-------------+
              |
              v
       Triage Engine
              |
       +------+------+
       |      |      |
       v      v      v
     Vitals  Trend  Waiting Time
       |      |      |
       +------+------+
              |
              v
       Priority Queue
              |
              v
           Supabase
```

## Why ChestScan Is Different

Traditional image classification systems generally provide a prediction and confidence score.

ChestScan extends this workflow by connecting image analysis with clinical prioritization support.

The system combines:

* Chest X ray validation
* Pneumonia classification
* Explainable AI using Grad CAM
* Patient identification
* Vital sign integration
* Patient deterioration tracking
* Emergency level assignment
* Multi patient priority ordering
* Waiting time tracking
* Patient history
* Live triage queue

This creates a unified workflow from medical image analysis to patient prioritization support.

## Safety and Clinical Disclaimer

ChestScan is a decision support system.

It does not replace a radiologist, physician, nurse, or other qualified healthcare professional.

AI predictions, Grad CAM visualizations, vital sign scores, deterioration trends, and triage priorities should be reviewed by qualified clinical staff before making medical decisions.

The system is intended to support clinical workflow rather than independently diagnose or determine treatment.

## Future Improvements

Potential future developments include:

* Integration with hospital information systems
* Real time wearable vital sign integration
* Additional respiratory diseases
* Multi disease chest X ray classification
* Automated clinical report generation
* Improved longitudinal patient monitoring
* More advanced deterioration prediction
* Multi hospital deployment
* Role based healthcare access
* Audit logs for clinical decisions
* Model performance monitoring
* External validation using independent clinical datasets

## Conclusion

ChestScan combines deep learning based chest X ray analysis with explainable AI and emergency triage support.

Instead of stopping at pneumonia classification, the system connects the prediction with patient vitals, clinical trends, waiting time, and patient history to organize a dynamic triage queue.

Its primary purpose is to provide healthcare teams with interpretable AI assisted information that can support faster and more structured patient prioritization while keeping final clinical decisions with qualified professionals.
