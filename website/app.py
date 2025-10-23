from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image, ImageStat
import io
import os
from datetime import datetime
import cv2
import numpy as np
import base64

app = Flask(__name__, static_folder='frontend')
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to /website
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'fundus_model_best.pth'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
SEVERITY_DESCRIPTIONS = {
    0: 'No Diabetic Retinopathy detected. Regular eye check-ups recommended.',
    1: 'Mild Non-Proliferative DR. Early stage with microaneurysms.',
    2: 'Moderate Non-Proliferative DR. More widespread damage to retina.',
    3: 'Severe Non-Proliferative DR. Significant blood vessel blockage.',
    4: 'Proliferative DR. Advanced stage requiring immediate attention.'
}

# XAI Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high_confidence': 85.0,      # > 85% - High confidence
    'moderate_confidence': 65.0,  # 65-85% - Moderate confidence
    'low_confidence': 65.0        # < 65% - Low confidence, flag for review
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'blur_threshold': 100.0,
    'min_brightness': 30,
    'max_brightness': 225,
    'min_contrast': 20,
    'artifact_threshold': 0.15,
    'centering_tolerance': 0.3
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM Implementation
class GradCAM:
    """Gradient-weighted Class Activation Mapping for model explainability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Class Activation Map"""
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        weighted_activations = pooled_gradients * self.activations
        cam = torch.sum(weighted_activations, dim=1).squeeze()
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class

def create_heatmap_overlay(original_image_pil, cam, alpha=0.5):
    """Create heatmap overlay on original image"""
    # Resize CAM to original image size
    original_image = np.array(original_image_pil)
    h, w = original_image.shape[:2]
    
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert to heatmap
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay on original image
    overlayed = cv2.addWeighted(original_image, 1-alpha, heatmap, alpha, 0)
    
    return overlayed

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    # Convert to PIL Image
    img = Image.fromarray(image_array.astype('uint8'))
    
    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def assess_prediction_confidence(confidence, probabilities):
    """Assess confidence level and determine if review is needed"""
    confidence_assessment = {
        'confidence_level': '',
        'requires_review': False,
        'review_reason': '',
        'recommendation': ''
    }
    
    # Check confidence level
    if confidence >= CONFIDENCE_THRESHOLDS['high_confidence']:
        confidence_assessment['confidence_level'] = 'high'
        confidence_assessment['recommendation'] = 'High confidence prediction. Suitable for automated screening.'
    elif confidence >= CONFIDENCE_THRESHOLDS['moderate_confidence']:
        confidence_assessment['confidence_level'] = 'moderate'
        confidence_assessment['recommendation'] = 'Moderate confidence. Consider specialist review for confirmation.'
        confidence_assessment['requires_review'] = True
        confidence_assessment['review_reason'] = 'Moderate confidence level'
    else:
        confidence_assessment['confidence_level'] = 'low'
        confidence_assessment['requires_review'] = True
        confidence_assessment['review_reason'] = 'Low confidence prediction'
        confidence_assessment['recommendation'] = 'Low confidence. Refer to specialist for expert review.'
    
    # Check for ambiguous cases (similar probabilities)
    sorted_probs = sorted(probabilities, reverse=True)
    if len(sorted_probs) >= 2:
        prob_diff = sorted_probs[0] - sorted_probs[1]
        if prob_diff < 0.15:  # Less than 15% difference
            confidence_assessment['requires_review'] = True
            confidence_assessment['review_reason'] = 'Ambiguous case: Similar probabilities for multiple classes'
            confidence_assessment['recommendation'] = 'Uncertain diagnosis. Multiple severity levels possible. Specialist review required.'
    
    return confidence_assessment

def assess_image_quality(image_pil):
    """Comprehensive image quality assessment"""
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    quality_report = {
        'overall_quality': 'good',
        'quality_score': 100,
        'issues': [],
        'warnings': [],
        'recommendations': [],
        'metrics': {}
    }
    
    # Blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_report['metrics']['sharpness'] = round(laplacian_var, 2)
    
    if laplacian_var < QUALITY_THRESHOLDS['blur_threshold']:
        quality_report['issues'].append('Image is blurry')
        quality_report['recommendations'].append('Please refocus the camera and retake the image')
        quality_report['quality_score'] -= 30
        quality_report['overall_quality'] = 'poor'
    elif laplacian_var < QUALITY_THRESHOLDS['blur_threshold'] * 1.5:
        quality_report['warnings'].append('Image sharpness is marginal')
        quality_report['recommendations'].append('Consider retaking with better focus')
        quality_report['quality_score'] -= 10
        if quality_report['overall_quality'] == 'good':
            quality_report['overall_quality'] = 'acceptable'
    
    # Illumination
    mean_brightness = np.mean(gray)
    quality_report['metrics']['brightness'] = round(mean_brightness, 2)
    
    if mean_brightness < QUALITY_THRESHOLDS['min_brightness']:
        quality_report['issues'].append('Image is too dark')
        quality_report['recommendations'].append('Increase illumination or adjust camera exposure')
        quality_report['quality_score'] -= 25
        quality_report['overall_quality'] = 'poor'
    elif mean_brightness > QUALITY_THRESHOLDS['max_brightness']:
        quality_report['issues'].append('Image is overexposed')
        quality_report['recommendations'].append('Reduce illumination or adjust camera exposure')
        quality_report['quality_score'] -= 25
        quality_report['overall_quality'] = 'poor'
    
    # Contrast
    contrast = gray.std()
    quality_report['metrics']['contrast'] = round(contrast, 2)
    
    if contrast < QUALITY_THRESHOLDS['min_contrast']:
        quality_report['issues'].append('Image has low contrast')
        quality_report['recommendations'].append('Adjust lighting conditions to improve image contrast')
        quality_report['quality_score'] -= 15
        if quality_report['overall_quality'] == 'good':
            quality_report['overall_quality'] = 'acceptable'
    
    # Artifacts
    black_pixels = np.sum(gray < 20)
    black_ratio = black_pixels / (height * width)
    quality_report['metrics']['artifact_ratio'] = round(black_ratio, 4)
    
    if black_ratio > QUALITY_THRESHOLDS['artifact_threshold']:
        quality_report['issues'].append('Image contains artifacts (eyelashes, dust, or obstruction)')
        quality_report['recommendations'].append('Clean the camera lens and ensure no obstruction')
        quality_report['quality_score'] -= 20
        if quality_report['overall_quality'] != 'poor':
            quality_report['overall_quality'] = 'acceptable'
    
    # Resolution
    quality_report['metrics']['resolution'] = f"{width}x{height}"
    if width < 500 or height < 500:
        quality_report['warnings'].append('Image resolution is low')
        quality_report['recommendations'].append('Use a higher resolution camera if possible')
        quality_report['quality_score'] -= 5
    
    quality_report['quality_score'] = max(0, quality_report['quality_score'])
    
    if quality_report['quality_score'] >= 80:
        quality_report['overall_quality'] = 'excellent'
    elif quality_report['quality_score'] >= 60:
        quality_report['overall_quality'] = 'good'
    elif quality_report['quality_score'] >= 40:
        quality_report['overall_quality'] = 'acceptable'
    else:
        quality_report['overall_quality'] = 'poor'
    
    quality_report['suitable_for_analysis'] = quality_report['quality_score'] >= 40
    
    return quality_report

def load_model():
    """Load EfficientNet-B0 model"""
    print(f"\n{'='*50}")
    print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
    print(f"Device: {DEVICE}")
    print(f"{'='*50}\n")
    
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Model loaded successfully!")
            
            first_param = next(model.parameters())
            print(f"✓ Weight check - Mean: {first_param.mean():.6f}, Std: {first_param.std():.6f}")
            
            if first_param.std() > 0.01:
                print("✓ Model weights verified!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    else:
        print(f"✗ Model file not found: {MODEL_PATH}")
    
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        dummy_output = model(dummy_input)
        print(f"✓ Model warmup complete\n")
    
    print(f"{'='*50}\n")
    return model

model = load_model()

# Initialize Grad-CAM
# For EfficientNet, use the last convolutional block
try:
    target_layer = model.conv_head if hasattr(model, 'conv_head') else model.blocks[-1]
    gradcam = GradCAM(model, target_layer)
    print("✓ Grad-CAM initialized successfully\n")
except Exception as e:
    print(f"⚠ Grad-CAM initialization warning: {e}\n")
    gradcam = None

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists(MODEL_PATH),
        'gradcam_enabled': gradcam is not None,
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict with XAI explanations and quality assessment"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_size = image.size
        
        print(f"\n--- Processing: {file.filename} ---")
        
        # Quality assessment
        quality_report = assess_image_quality(image)
        print(f"Quality Score: {quality_report['quality_score']}/100")
        
        if not quality_report['suitable_for_analysis']:
            return jsonify({
                'success': False,
                'error': 'Image quality is too poor for reliable analysis',
                'quality_report': quality_report,
                'recommendation': 'Please improve image quality and try again'
            }), 400
        
        # Prepare input
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item() * 100
            all_probabilities = probabilities[0].cpu().numpy()
            
            print(f"Predicted: Class {predicted_class} ({CLASS_NAMES[predicted_class]}) @ {confidence:.2f}%")
        
        # Confidence assessment
        confidence_assessment = assess_prediction_confidence(confidence, all_probabilities)
        print(f"Confidence Level: {confidence_assessment['confidence_level']}")
        print(f"Requires Review: {confidence_assessment['requires_review']}")
        
        # Generate Grad-CAM heatmap
        gradcam_data = None
        if gradcam is not None:
            try:
                cam, _ = gradcam.generate_cam(input_tensor, predicted_class)
                heatmap_overlay = create_heatmap_overlay(image, cam, alpha=0.4)
                
                # Also create pure heatmap
                cam_resized = cv2.resize(cam, (original_size[0], original_size[1]))
                heatmap_pure = np.uint8(255 * cam_resized)
                heatmap_pure = cv2.applyColorMap(heatmap_pure, cv2.COLORMAP_JET)
                heatmap_pure = cv2.cvtColor(heatmap_pure, cv2.COLOR_BGR2RGB)
                
                gradcam_data = {
                    'overlay': image_to_base64(heatmap_overlay),
                    'heatmap': image_to_base64(heatmap_pure),
                    'description': 'Red regions indicate areas that most strongly influenced the AI\'s decision'
                }
                
                print("✓ Grad-CAM heatmap generated")
            except Exception as e:
                print(f"⚠ Grad-CAM generation failed: {e}")
        
        response = {
            'success': True,
            'prediction': {
                'class': int(predicted_class),
                'class_name': CLASS_NAMES[predicted_class],
                'confidence': float(confidence),
                'description': SEVERITY_DESCRIPTIONS[predicted_class]
            },
            'probabilities': [
                {
                    'class': i,
                    'class_name': CLASS_NAMES[i],
                    'probability': float(all_probabilities[i] * 100)
                }
                for i in range(5)
            ],
            'confidence_assessment': confidence_assessment,
            'explainability': gradcam_data,
            'quality_report': quality_report,
            'image_info': {
                'original_size': original_size,
                'processed_size': (224, 224)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print()
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction with quality checks"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if len(files) == 0 or len(files) > 10:
            return jsonify({'error': 'Provide 1-10 images'}), 400
        
        results = []
        for file in files:
            try:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
                quality_report = assess_image_quality(image)
                
                if not quality_report['suitable_for_analysis']:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': 'Poor image quality',
                        'quality_score': quality_report['quality_score']
                    })
                    continue
                
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    all_probabilities = probabilities[0].cpu().numpy()
                
                confidence_assessment = assess_prediction_confidence(
                    confidence.item() * 100, 
                    all_probabilities
                )
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'class': int(predicted_class.item()),
                    'class_name': CLASS_NAMES[predicted_class.item()],
                    'confidence': float(confidence.item() * 100),
                    'requires_review': confidence_assessment['requires_review'],
                    'quality_score': quality_report['quality_score']
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(files),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  FUNDUS IMAGE CLASSIFICATION SERVER")
    print("  with Quality Assessment & Explainable AI")
    print("="*60)
    print(f"  http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
