// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const closePreview = document.getElementById('closePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const analyzeBtnText = document.getElementById('analyzeBtnText');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const downloadResultsBtn = document.getElementById('downloadResultsBtn');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');

// State
let selectedFile = null;
let currentResults = null;
let analysisStartTime = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkServerHealth();
});

// Event Listeners
function setupEventListeners() {
    // Mobile menu
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', toggleMobileMenu);
    }
    
    // Browse button
    if (browseBtn) {
        browseBtn.addEventListener('click', () => fileInput.click());
    }
    
    // File input
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Drag and drop
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }
    
    // Preview controls
    if (closePreview) {
        closePreview.addEventListener('click', resetUpload);
    }
    
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeImage);
    }
    
    // Results controls
    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', resetUpload);
    }
    
    if (downloadResultsBtn) {
        downloadResultsBtn.addEventListener('click', downloadResults);
    }
}

// Mobile Menu Toggle
function toggleMobileMenu() {
    mobileMenuBtn.classList.toggle('active');
    const mainNav = document.querySelector('.main-nav');
    if (mainNav) {
        mainNav.classList.toggle('mobile-open');
    }
}

// File Selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) processFile(file);
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = event.dataTransfer.files[0];
    if (file) processFile(file);
}

function processFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload PNG, JPG, or JPEG images only.');
        return;
    }
    
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 10MB. Please upload a smaller image.');
        return;
    }
    
    selectedFile = file;
    displayPreview(file);
}

function displayPreview(file) {
    uploadArea.style.display = 'none';
    previewContainer.style.display = 'block';
    errorMessage.style.display = 'none';
    resultsSection.style.display = 'none';
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    fileName.textContent = `${file.name}`;
    fileSize.textContent = `${formatFileSize(file.size)}`;
}

// Image Analysis

// Image Analysis with Retina Scan Loader
// Image Analysis with Retina Scan Loader - WITH NULL CHECKS
async function analyzeImage() {
    if (!selectedFile) return;
    
    // Safely access elements with null checks
    if (analyzeBtn) analyzeBtn.disabled = true;
    if (analyzeBtnText) analyzeBtnText.style.display = 'none';
    if (errorMessage) errorMessage.style.display = 'none';
    
    // Show retina scan loader
    const retinaScanLoader = document.getElementById('retinaScanLoader');
    const scanStep = document.getElementById('scanStep');
    
    if (retinaScanLoader) {
        retinaScanLoader.style.display = 'flex';
        
        // Simulate analysis steps with dynamic text
        const steps = [
            'Initializing AI model',
            'Processing image quality',
            'Detecting retinal features',
            'Analyzing severity patterns',
            'Generating explanation'
        ];
        
        let currentStep = 0;
        const stepInterval = setInterval(() => {
            if (currentStep < steps.length - 1 && scanStep) {
                currentStep++;
                scanStep.textContent = steps[currentStep];
            }
        }, 800);
        
        // Clear interval after animation completes
        setTimeout(() => clearInterval(stepInterval), 4000);
    }
    
    analysisStartTime = Date.now();
    
    try {
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Hide loader with smooth transition and null check
        if (retinaScanLoader) {
            setTimeout(() => {
                retinaScanLoader.style.opacity = '0';
                setTimeout(() => {
                    retinaScanLoader.style.display = 'none';
                    retinaScanLoader.style.opacity = '1';
                }, 300);
            }, 500);
        }
        
        if (response.ok && data.success) {
            currentResults = data;
            displayResults(data);
        } else {
            if (data.quality_report) {
                showQualityError(data);
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        
        // Hide loader on error with null check
        if (retinaScanLoader) {
            retinaScanLoader.style.display = 'none';
        }
        
        showError(`Analysis failed: ${error.message}`);
        
        // Reset button states with null checks
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (analyzeBtnText) analyzeBtnText.style.display = 'block';
    }
}




// Display Quality Error
function showQualityError(data) {
    previewContainer.style.display = 'none';
    resultsSection.style.display = 'block';
    
    // Hide other sections
    const sections = document.querySelectorAll('.result-section');
    sections.forEach(section => {
        if (!section.querySelector('.quality-overview')) {
            section.style.display = 'none';
        }
    });
    
    const explainabilitySection = document.getElementById('explainabilitySection');
    if (explainabilitySection) {
        explainabilitySection.style.display = 'none';
    }
    
    const confidenceAlert = document.getElementById('confidenceAlert');
    if (confidenceAlert) {
        confidenceAlert.style.display = 'none';
    }
    
    displayQualityReport(data.quality_report);
    showError(data.error + '. Please check the quality recommendations below.');
    
    analyzeBtn.disabled = false;
    analyzeBtnText.style.display = 'block';
    loadingSpinner.style.display = 'none';
}

// Display Results
// Display Results
function displayResults(data) {
    const analysisTime = ((Date.now() - analysisStartTime) / 1000).toFixed(2);
    const prediction = data.prediction;
    
    // Hide preview and show results with null checks
    if (previewContainer) previewContainer.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'block';
    
    // Show all sections with null check
    const sections = document.querySelectorAll('.result-section');
    sections.forEach(section => {
        if (section) section.style.display = 'block';
    });
    
    // Display Confidence Alert
    if (data.confidence_assessment) {
        displayConfidenceAlert(data.confidence_assessment);
    }
    
    // Display Explainability (Grad-CAM)
    if (data.explainability) {
        displayExplainability(data.explainability, prediction);
    }
    
    // Display Quality Report
    if (data.quality_report) {
        displayQualityReport(data.quality_report);
    }
    
    // Severity Badge
    const severityBadge = document.getElementById('severityBadge');
    const severityClass = document.getElementById('severityClass');
    if (severityBadge && severityClass) {
        severityBadge.className = `severity-badge severity-${prediction.class}`;
        severityClass.textContent = prediction.class_name;
    }
    
    // Confidence
    const confidenceValue = document.getElementById('confidenceValue');
    if (confidenceValue) {
        confidenceValue.textContent = `${prediction.confidence.toFixed(1)}%`;
    }
    
    // Confidence Level Badge
    if (data.confidence_assessment) {
        displayConfidenceLevelBadge(data.confidence_assessment);
    }
    
    // Description
    const predictionDescription = document.getElementById('predictionDescription');
    if (predictionDescription) {
        predictionDescription.textContent = prediction.description;
    }
    
    // Probability Bars
    const probabilityBars = document.getElementById('probabilityBars');
    if (probabilityBars) {
        probabilityBars.innerHTML = '';
        data.probabilities.forEach(prob => {
            const barHTML = `
                <div class="probability-bar">
                    <div class="probability-label">
                        <span class="probability-name">${prob.class_name}</span>
                        <span class="probability-value">${prob.probability.toFixed(1)}%</span>
                    </div>
                    <div class="probability-track">
                        <div class="probability-fill" style="width: ${prob.probability}%"></div>
                    </div>
                </div>
            `;
            probabilityBars.innerHTML += barHTML;
        });
    }
    
    // Additional Info
    const predictedClassName = document.getElementById('predictedClassName');
    if (predictedClassName) {
        predictedClassName.textContent = prediction.class_name;
    }
    
    const imageSize = document.getElementById('imageSize');
    if (imageSize && data.image_info) {
        imageSize.textContent = `${data.image_info.original_size[0]} × ${data.image_info.original_size[1]} px`;
    }
    
    const analysisTimeEl = document.getElementById('analysisTime');
    if (analysisTimeEl) {
        analysisTimeEl.textContent = `${analysisTime}s`;
    }
    
    // Scroll to results
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Reset analyze button state
    if (analyzeBtn) analyzeBtn.disabled = false;
    if (analyzeBtnText) analyzeBtnText.style.display = 'block';
}


// Display Confidence Alert
function displayConfidenceAlert(assessment) {
    const alertBox = document.getElementById('confidenceAlert');
    if (!alertBox) return;
    
    if (assessment.requires_review) {
        alertBox.style.display = 'flex';
        
        let alertClass = '';
        let icon = '';
        
        if (assessment.confidence_level === 'low') {
            alertClass = 'alert-danger';
            icon = '🚨';
        } else {
            alertClass = 'alert-warning';
            icon = '⚠️';
        }
        
        alertBox.className = `confidence-alert ${alertClass}`;
        alertBox.innerHTML = `
            <div class="alert-icon">${icon}</div>
            <div class="alert-content">
                <h4>Specialist Review Required</h4>
                <p><strong>Reason:</strong> ${assessment.review_reason}</p>
                <p>${assessment.recommendation}</p>
            </div>
        `;
    } else {
        alertBox.style.display = 'none';
    }
}

// Display Explainability (Grad-CAM)
function displayExplainability(explainability, prediction) {
    const xaiSection = document.getElementById('explainabilitySection');
    if (!xaiSection) return;
    
    xaiSection.style.display = 'block';
    
    const originalImage = document.getElementById('originalImage');
    if (originalImage) {
        originalImage.src = imagePreview.src;
    }
    
    const heatmapOverlay = document.getElementById('heatmapOverlay');
    if (heatmapOverlay) {
        heatmapOverlay.src = explainability.overlay;
    }
}

// Display Confidence Level Badge
function displayConfidenceLevelBadge(assessment) {
    const badge = document.getElementById('confidenceLevelBadge');
    if (!badge) return;
    
    let badgeClass = '';
    let badgeText = '';
    let badgeIcon = '';
    
    switch(assessment.confidence_level) {
        case 'high':
            badgeClass = 'badge-high';
            badgeText = 'High Confidence';
            badgeIcon = '✓';
            break;
        case 'moderate':
            badgeClass = 'badge-moderate';
            badgeText = 'Moderate Confidence';
            badgeIcon = '~';
            break;
        case 'low':
            badgeClass = 'badge-low';
            badgeText = 'Low Confidence';
            badgeIcon = '!';
            break;
    }
    
    badge.className = `confidence-level-badge ${badgeClass}`;
    badge.innerHTML = `<span class="badge-icon">${badgeIcon}</span> ${badgeText}`;
}

// Display Quality Report
function displayQualityReport(quality_report) {
    const scoreCircle = document.getElementById('qualityScoreCircle');
    const scoreNumber = document.getElementById('qualityScoreNumber');
    
    if (scoreNumber) {
        scoreNumber.textContent = quality_report.quality_score;
    }
    
    if (scoreCircle) {
        if (quality_report.quality_score >= 80) {
            scoreCircle.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
        } else if (quality_report.quality_score >= 60) {
            scoreCircle.style.background = 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)';
        } else if (quality_report.quality_score >= 40) {
            scoreCircle.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
        } else {
            scoreCircle.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
        }
    }
    
    const qualityStatus = document.getElementById('qualityStatus');
    if (qualityStatus) {
        const statusClass = `status-${quality_report.overall_quality}`;
        qualityStatus.innerHTML = `
            <div class="quality-status-badge ${statusClass}">
                ${quality_report.overall_quality.toUpperCase()}
            </div>
            <p style="color: var(--text-secondary); margin-top: 8px;">
                ${quality_report.suitable_for_analysis ? 
                  '✓ Image is suitable for analysis' : 
                  '✗ Image quality needs improvement'}
            </p>
        `;
    }
    
    const qualityMetrics = document.getElementById('qualityMetrics');
    if (qualityMetrics) {
        qualityMetrics.innerHTML = '';
        
        const metrics = [
            { label: 'Sharpness', value: quality_report.metrics.sharpness },
            { label: 'Brightness', value: quality_report.metrics.brightness },
            { label: 'Contrast', value: quality_report.metrics.contrast },
            { label: 'Resolution', value: quality_report.metrics.resolution }
        ];
        
        metrics.forEach(metric => {
            qualityMetrics.innerHTML += `
                <div class="quality-metric">
                    <div class="metric-label">${metric.label}</div>
                    <div class="metric-value">${metric.value}</div>
                </div>
            `;
        });
    }
    
    const qualityIssues = document.getElementById('qualityIssues');
    const issuesList = document.getElementById('issuesList');
    
    if (quality_report.issues && quality_report.issues.length > 0) {
        if (qualityIssues) qualityIssues.style.display = 'block';
        if (issuesList) {
            issuesList.innerHTML = '';
            quality_report.issues.forEach(issue => {
                issuesList.innerHTML += `<li>${issue}</li>`;
            });
        }
    } else {
        if (qualityIssues) qualityIssues.style.display = 'none';
    }
    
    const qualityRecommendations = document.getElementById('qualityRecommendations');
    const recommendationsList = document.getElementById('recommendationsList');
    
    if (quality_report.recommendations && quality_report.recommendations.length > 0) {
        if (qualityRecommendations) qualityRecommendations.style.display = 'block';
        if (recommendationsList) {
            recommendationsList.innerHTML = '';
            quality_report.recommendations.forEach(rec => {
                recommendationsList.innerHTML += `<li>${rec}</li>`;
            });
        }
    } else {
        if (qualityRecommendations) qualityRecommendations.style.display = 'none';
    }
}

// Reset Upload
function resetUpload() {
    selectedFile = null;
    currentResults = null;
    fileInput.value = '';
    
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    
    analyzeBtn.disabled = false;
    analyzeBtnText.style.display = 'block';
    loadingSpinner.style.display = 'none';
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Download Results
function downloadResults() {
    if (!currentResults) return;
    
    const results = {
        timestamp: currentResults.timestamp,
        prediction: currentResults.prediction,
        probabilities: currentResults.probabilities,
        confidence_assessment: currentResults.confidence_assessment,
        quality_report: currentResults.quality_report,
        image_info: currentResults.image_info
    };
    
    const blob = new Blob([JSON.stringify(results, null, 2)], 
        { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fundus_analysis_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    if (!errorMessage) return; // Exit if element doesn't exist
    
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    setTimeout(() => {
        if (errorMessage) {
            errorMessage.style.display = 'none';
        }
    }, 8000);
}


async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('✓ Server Status:', data);
        
        if (data.status === 'healthy') {
            console.log('✓ Model Loaded:', data.model_loaded);
            console.log('✓ Grad-CAM Enabled:', data.gradcam_enabled);
        }
    } catch (error) {
        console.error('✗ Server health check failed:', error);
        showError('Unable to connect to server. Please ensure the backend is running on port 5000.');
    }
}
