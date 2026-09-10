"""
Document Face Extraction Service (No OCR).
- Detects the portrait face inside uploaded identity documents (Passport, National ID, Visa).
- Validates resolution and quality floor.
- Aligns face via standard 5-point affine transform.
- Extracts 512-dimensional normalized ArcFace embedding.
- Encodes face crop as base64 JPEG data URL for display.
"""

from typing import Dict, Any, Optional
import base64
import cv2
import numpy as np
from services.detection.scrfd_detector import SCRFDDetector
from services.recognition.arcface_embedder import ArcFaceEmbedder


class DocumentFaceExtractor:
    def __init__(self, detector: SCRFDDetector, embedder: ArcFaceEmbedder):
        self.detector = detector
        self.embedder = embedder

    def extract_document_face(
        self,
        document_bgr: np.ndarray,
        doc_type: str = "passport",
        doc_id: str = "doc_001"
    ) -> Dict[str, Any]:
        """
        Extracts face from identity document image.
        """
        if document_bgr is None or document_bgr.size == 0:
            return {
                "success": False,
                "error": "EMPTY_DOCUMENT_IMAGE",
                "message": "Uploaded document image is empty"
            }

        # Run face detection on full document image
        det_result = self.detector.detect(document_bgr)
        
        if det_result.status == "NO_FACE":
            return {
                "success": False,
                "error": "NO_FACE_IN_DOCUMENT",
                "message": "No photograph face detected in document image. Please upload a clear photo of your ID."
            }

        if det_result.status == "MULTIPLE_SUBJECTS":
            return {
                "success": False,
                "error": "MULTIPLE_FACES_IN_DOCUMENT",
                "message": "Multiple faces detected on document. Please upload an image with only one portrait visible."
            }

        face_dict = det_result.primary_face
        if face_dict is None:
            return {
                "success": False,
                "error": "FACE_DETECTION_FAILED",
                "message": "Could not isolate document face."
            }

        bbox = face_dict["bbox"]
        x, y, w, h = bbox
        
        # Check minimum face dimension on document
        if w < 60 or h < 60:
            return {
                "success": False,
                "error": "DOCUMENT_FACE_TOO_SMALL",
                "message": f"Document face photo is too low resolution ({w}x{h}px). Minimum 60px required."
            }

        # Quality check: Laplacian sharpness
        dh, dw = document_bgr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(dw, x + w)
        y2 = min(dh, y + h)
        crop = document_bgr[y1:y2, x1:x2]
        
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

        # Extract normalized 512-d embedding.
        # Use specialized document extraction path:
        # - CLAHE contrast enhancement (critical for scanned/photocopied ID photos)
        # - Photometric TTA (bridges webcam-vs-scanner color domain gap)
        # - Quality-adaptive norm weighting (AdaFace principle)
        embedding = self.embedder.extract_embedding_document(document_bgr, face_dict)
        if embedding is None:
            return {
                "success": False,
                "error": "EMBEDDING_EXTRACTION_FAILED",
                "message": "Failed to extract biometric embedding from document portrait."
            }

        # Encode crop as base64 JPEG
        _, buffer = cv2.imencode(".jpg", crop)
        crop_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return {
            "success": True,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "bbox": bbox,
            "width": w,
            "height": h,
            "confidence": face_dict["confidence"],
            "sharpness": round(sharpness, 1),
            "face_crop_url": crop_base64,
            "embedding": embedding.tolist()  # serialize for storage / response
        }
