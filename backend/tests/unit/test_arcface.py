"""
Unit test for ArcFace Biometric Embedder and Cosine Similarity.
"""

import cv2
import numpy as np
from services.detection.scrfd_detector import SCRFDDetector
from services.recognition.arcface_embedder import ArcFaceEmbedder


def test_arcface_embedding_properties():
    img = cv2.imread("videoframe_4589.png")
    detector = SCRFDDetector(min_confidence=0.60)
    det_res = detector.detect(img)
    face = det_res.primary_face
    assert face is not None

    embedder = ArcFaceEmbedder()
    emb = embedder.extract_embedding(img, face)
    assert emb is not None
    assert emb.shape == (512,)
    
    # Check L2 normalization
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4

    # Self similarity must be 1.0
    self_sim = embedder.cosine_similarity(emb, emb)
    assert abs(self_sim - 1.0) < 1e-4

    # Random noise embedding should have low similarity (< 0.3)
    random_emb = np.random.randn(512).astype(np.float32)
    random_emb /= np.linalg.norm(random_emb)
    rand_sim = embedder.cosine_similarity(emb, random_emb)
    assert rand_sim < 0.35
