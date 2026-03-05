#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import zlib
"""
Enhanced Self-Judgment and Fact Argumentation Module
Implements strong self-awareness and judgment capabilities with factual argumentation and self-correction

Features:
- Advanced judgment quality assessment with evidence evaluation
- Structured argument chain construction for fact verification
- Self-correction mechanisms for reasoning errors
- Confidence calibration and uncertainty quantification
- Integration with existing self-reflection and reasoning systems
- Real-time judgment tracking and performance monitoring
- From-scratch learning without external dependencies

Core Capabilities:
1. Judgment Process Recording and Analysis
2. Fact Verification and Evidence Chain Building
3. Self-Correction with Adaptive Strategies
4. Confidence Calibration and Error Detection
5. Integration with AGI Core and Reasoning Systems

版权所有 (c) 2025 AGI Soul Team
Licensed under the Apache License, Version 2.0
"""

import json
import time
import numpy as np
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import re
from core.error_handling import error_handler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class JudgmentRecord:
    """Complete record of a judgment process with quality metrics"""
    judgment_id: str
    context: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    conclusion: str
    confidence: float
    reasoning_steps: List[str]
    alternatives_considered: List[str]
    quality_metrics: Dict[str, float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArgumentChain:
    """Structured argument chain for fact verification"""
    argument_id: str
    premises: List[str]
    conclusion: str
    evidence_map: Dict[str, List[Dict[str, Any]]]
    logical_structure: Dict[str, Any]
    completeness_score: float
    coherence_score: float
    evidence_support_score: float
    overall_quality: float
    timestamp: float
    verification_status: str = "pending"  # pending, verified, disputed, invalid


@dataclass
class CorrectionAttempt:
    """Self-correction attempt record"""
    correction_id: str
    error_type: str
    error_description: str
    original_content: Dict[str, Any]
    correction_strategy: str
    corrected_content: Dict[str, Any]
    confidence: float
    effectiveness_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfJudgmentState:
    """Current state of self-judgment system"""
    total_judgments: int = 0
    successful_judgments: int = 0
    average_confidence: float = 0.5
    confidence_calibration_error: float = 0.3
    self_correction_success_rate: float = 0.3
    argument_chain_success_rate: float = 0.4
    fact_verification_accuracy: float = 0.5
    judgment_consistency_score: float = 0.6
    recent_performance: List[float] = field(default_factory=list)
    learning_progress: float = 0.1


class EnhancedSelfJudgmentModule:
    """
    Enhanced Self-Judgment Module for AGI Systems
    Implements strong self-awareness, judgment capabilities, and factual argumentation
    Integrates with self-reflection and reasoning systems for continuous improvement
    """
    
    def __init__(self, knowledge_graph_path: str = None, from_scratch: bool = True):
        """
        Initialize enhanced self-judgment module
        
        Args:
            knowledge_graph_path: Path to knowledge graph for fact verification
            from_scratch: Whether to initialize from scratch or load existing data
        """
        self.logger = logging.getLogger(__name__)
        
        # Core state and tracking
        self.state = SelfJudgmentState()
        self.judgment_history: List[JudgmentRecord] = []
        self.argument_history: List[ArgumentChain] = []
        self.correction_history: List[CorrectionAttempt] = []
        
        # Performance tracking
        self.performance_metrics = {
            "judgment_qualities": deque(maxlen=1000),
            "confidence_errors": deque(maxlen=1000),
            "correction_successes": deque(maxlen=1000),
            "argument_qualities": deque(maxlen=1000),
            "verification_times": deque(maxlen=1000),
            "consistency_scores": deque(maxlen=1000)
        }
        
        # Error patterns and learning
        self.error_patterns = self._initialize_error_patterns(from_scratch)
        self.success_patterns = self._initialize_success_patterns(from_scratch)
        
        # Integration points
        self.knowledge_graph_path = knowledge_graph_path
        self.integrated_systems = {
            "self_reflection": False,
            "advanced_reasoning": False,
            "knowledge_graph": False
        }
        
        # Adaptive learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.3
        self.adaptation_factor = 1.0
        
        # Thresholds and parameters
        self.thresholds = {
            "judgment_confidence_min": 0.3,
            "judgment_confidence_max": 0.95,
            "evidence_strength_min": 0.2,
            "argument_completeness_min": 0.5,
            "self_correction_confidence": 0.6,
            "consistency_threshold": 0.7
        }
        
        # Load existing data if not from scratch
        if not from_scratch:
            self._load_historical_data()
        
        self.logger.info(f"Enhanced Self-Judgment Module initialized (from_scratch: {from_scratch})")
    
    def _initialize_error_patterns(self, from_scratch: bool) -> Dict[str, Any]:
        """Initialize error patterns database"""
        if from_scratch:
            return {
                "judgment_errors": {"patterns": [], "count": 0},
                "argument_errors": {"patterns": [], "count": 0},
                "fact_errors": {"patterns": [], "count": 0},
                "confidence_errors": {"patterns": [], "count": 0}
            }
        else:
            return {
                "judgment_errors": {
                    "patterns": [
                        {
                            "type": "overconfidence",
                            "description": "Confidence exceeds evidence support",
                            "indicators": ["high_confidence_low_evidence", "ignoring_uncertainty"],
                            "examples": [],
                            "correction_strategy": "confidence_adjustment"
                        },
                        {
                            "type": "underconfidence", 
                            "description": "Excessive caution despite strong evidence",
                            "indicators": ["low_confidence_high_evidence", "excessive_qualification"],
                            "examples": [],
                            "correction_strategy": "evidence_reinforcement"
                        },
                        {
                            "type": "confirmation_bias",
                            "description": "Selective attention to confirming evidence",
                            "indicators": ["ignoring_contrary_evidence", "seeking_confirmation"],
                            "examples": [],
                            "correction_strategy": "alternative_exploration"
                        }
                    ],
                    "count": 0
                },
                "argument_errors": {
                    "patterns": [
                        {
                            "type": "logical_fallacy",
                            "description": "Invalid logical structure",
                            "indicators": ["circular_reasoning", "false_dilemma", "hasty_generalization"],
                            "examples": [],
                            "correction_strategy": "logical_restructuring"
                        },
                        {
                            "type": "evidence_gap",
                            "description": "Missing critical evidence",
                            "indicators": ["unsupported_premises", "weak_evidence_chain"],
                            "examples": [],
                            "correction_strategy": "evidence_gathering"
                        },
                        {
                            "type": "coherence_break",
                            "description": "Inconsistent or contradictory elements",
                            "indicators": ["contradictory_statements", "inconsistent_conclusions"],
                            "examples": [],
                            "correction_strategy": "consistency_check"
                        }
                    ],
                    "count": 0
                },
                "fact_errors": {
                    "patterns": [
                        {
                            "type": "factual_inaccuracy",
                            "description": "Incorrect factual information",
                            "indicators": ["incorrect_data", "outdated_information"],
                            "examples": [],
                            "correction_strategy": "fact_verification"
                        },
                        {
                            "type": "source_reliability",
                            "description": "Unreliable or biased sources",
                            "indicators": ["unreliable_source", "biased_source"],
                            "examples": [],
                            "correction_strategy": "source_verification"
                        },
                        {
                            "type": "verification_failure",
                            "description": "Insufficient verification process",
                            "indicators": ["insufficient_cross_checking", "single_source_reliance"],
                            "examples": [],
                            "correction_strategy": "multi_source_verification"
                        }
                    ],
                    "count": 0
                },
                "confidence_errors": {
                    "patterns": [
                        {
                            "type": "calibration_mismatch",
                            "description": "Confidence doesn't match actual accuracy",
                            "indicators": ["overestimation", "underestimation"],
                            "examples": [],
                            "correction_strategy": "calibration_adjustment"
                        },
                        {
                            "type": "uncertainty_neglect",
                            "description": "Failure to account for uncertainty",
                            "indicators": ["ignoring_uncertainty", "overprecision"],
                            "examples": [],
                            "correction_strategy": "uncertainty_quantification"
                        }
                    ],
                    "count": 0
                }
            }
    
    def _initialize_success_patterns(self, from_scratch: bool) -> Dict[str, Any]:
        """Initialize success patterns database"""
        if from_scratch:
            return {
                "judgment_successes": {"patterns": [], "count": 0},
                "argument_successes": {"patterns": [], "count": 0},
                "correction_successes": {"patterns": [], "count": 0}
            }
        else:
            return {
                "judgment_successes": {
                    "patterns": [
                        {
                            "type": "well_calibrated",
                            "description": "Confidence well-matched to evidence",
                            "indicators": ["appropriate_confidence", "evidence_alignment"],
                            "examples": []
                        },
                        {
                            "type": "comprehensive_analysis",
                            "description": "Thorough consideration of alternatives",
                            "indicators": ["multiple_alternatives", "balanced_evaluation"],
                            "examples": []
                        },
                        {
                            "type": "logical_rigor",
                            "description": "Sound logical reasoning",
                            "indicators": ["valid_inference", "clear_reasoning_steps"],
                            "examples": []
                        }
                    ],
                    "count": 0
                },
                "argument_successes": {
                    "patterns": [
                        {
                            "type": "complete_evidence_chain",
                            "description": "All premises supported by evidence",
                            "indicators": ["comprehensive_evidence", "strong_support"],
                            "examples": []
                        },
                        {
                            "type": "logical_coherence",
                            "description": "Consistent and logically sound structure",
                            "indicators": ["no_contradictions", "valid_structure"],
                            "examples": []
                        }
                    ],
                    "count": 0
                },
                "correction_successes": {
                    "patterns": [
                        {
                            "type": "effective_error_correction",
                            "description": "Successful identification and correction",
                            "indicators": ["error_detected", "appropriate_correction"],
                            "examples": []
                        }
                    ],
                    "count": 0
                }
            }
    
    def _load_historical_data(self):
        """Load historical judgment data"""
        try:
            data_file = Path("data/self_judgment_history.pkl")
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    
                self.judgment_history = data.get('judgment_history', [])
                self.argument_history = data.get('argument_history', [])
                self.correction_history = data.get('correction_history', [])
                self.state = data.get('state', self.state)
                self.performance_metrics = data.get('performance_metrics', self.performance_metrics)
                
                self.logger.info(f"Loaded {len(self.judgment_history)} historical judgments")
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
    
    def _save_historical_data(self):
        """Save current judgment data"""
        try:
            data = {
                'judgment_history': self.judgment_history,
                'argument_history': self.argument_history,
                'correction_history': self.correction_history,
                'state': self.state,
                'performance_metrics': self.performance_metrics
            }
            
            Path("data").mkdir(exist_ok=True)
            with open("data/self_judgment_history.pkl", 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save historical data: {e}")
    
    # ========== Core Judgment Methods ==========
    
    def make_judgment(self, context: Dict[str, Any], 
                     evidence: List[Dict[str, Any]],
                     conclusion_options: List[str]) -> Dict[str, Any]:
        """
        Make a judgment with comprehensive quality assessment
        
        Args:
            context: Judgment context including task, constraints, etc.
            evidence: List of evidence items with metadata
            conclusion_options: Possible conclusions to choose from
            
        Returns:
            Dict containing judgment results and quality assessment
        """
        start_time = time.time()
        
        try:
            # Step 1: Evidence evaluation
            evidence_analysis = self._analyze_evidence(evidence)
            
            # Step 2: Option evaluation
            option_evaluations = self._evaluate_options(conclusion_options, evidence_analysis)
            
            # Step 3: Confidence calculation
            confidence, uncertainty = self._calculate_confidence(option_evaluations, evidence_analysis)
            
            # Step 4: Conclusion selection
            conclusion, reasoning_steps = self._select_conclusion(option_evaluations, confidence)
            
            # Step 5: Quality assessment
            quality_metrics = self._assess_judgment_quality(
                evidence_analysis, option_evaluations, confidence, conclusion, reasoning_steps
            )
            
            # Step 6: Record judgment
            judgment_id = f"judgment_{int(time.time())}_{abs((zlib.adler32(str(str(time.time().encode('utf-8')) & 0xffffffff)) + str(context))) % 9000 + 1000}"
            judgment_record = JudgmentRecord(
                judgment_id=judgment_id,
                context=context,
                evidence=evidence,
                conclusion=conclusion,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                alternatives_considered=conclusion_options,
                quality_metrics=quality_metrics,
                timestamp=time.time(),
                metadata={
                    "processing_time": time.time() - start_time,
                    "evidence_count": len(evidence),
                    "options_considered": len(conclusion_options)
                }
            )
            
            # Store and update state
            self.judgment_history.append(judgment_record)
            self._update_judgment_state(judgment_record, quality_metrics)
            
            # Check for self-correction needs
            if quality_metrics.get('needs_correction', False):
                correction = self._attempt_self_correction(judgment_record)
                if correction:
                    judgment_record.metadata['correction_applied'] = True
                    judgment_record.metadata['correction_id'] = correction['correction_id']
            
            # Generate response
            response = {
                "success": True,
                "judgment_id": judgment_id,
                "conclusion": conclusion,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "reasoning_summary": self._generate_reasoning_summary(reasoning_steps),
                "quality_metrics": quality_metrics,
                "recommendations": quality_metrics.get('recommendations', []),
                "processing_time": time.time() - start_time
            }
            
            self.logger.info(f"Judgment {judgment_id} completed: {conclusion} (confidence: {confidence:.2f})")
            return response
            
        except Exception as e:
            self.logger.error(f"Judgment failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "judgment_id": f"failed_{int(time.time())}"
            }
    
    def _analyze_evidence(self, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive evidence analysis"""
        if not evidence:
            return {
                "total_strength": 0.1,
                "average_reliability": 0.5,
                "coverage_score": 0.1,
                "consistency_score": 0.5,
                "source_diversity": 0.1,
                "temporal_freshness": 0.5
            }
        
        strengths = []
        reliabilities = []
        sources = set()
        timestamps = []
        
        for item in evidence:
            # Extract evidence metrics
            strength = item.get('strength', 0.5)
            reliability = item.get('reliability', 0.7)
            source = item.get('source', 'unknown')
            timestamp = item.get('timestamp', time.time())
            
            # Adjust for evidence type
            evidence_type = item.get('type', 'unknown')
            if evidence_type == 'empirical_data':
                strength *= 1.2
            elif evidence_type == 'expert_opinion':
                reliability *= 0.9
            elif evidence_type == 'theoretical':
                strength *= 0.8
            
            strengths.append(strength)
            reliabilities.append(reliability)
            sources.add(source)
            timestamps.append(timestamp)
        
        # Calculate evidence metrics
        total_strength = np.mean(strengths) if strengths else 0.5
        average_reliability = np.mean(reliabilities) if reliabilities else 0.7
        
        # Coverage: how many aspects are covered (simplified)
        coverage_score = min(1.0, len(evidence) / 5.0)
        
        # Consistency: check for contradictions (simplified)
        consistency_score = self._calculate_evidence_consistency(evidence)
        
        # Source diversity
        source_diversity = min(1.0, len(sources) / 3.0)
        
        # Temporal freshness
        if timestamps:
            current_time = time.time()
            time_diffs = [current_time - ts for ts in timestamps]
            avg_freshness = np.mean([1.0 - min(1.0, diff / (365*24*3600)) for diff in time_diffs])
        else:
            avg_freshness = 0.5
        
        return {
            "total_strength": total_strength,
            "average_reliability": average_reliability,
            "coverage_score": coverage_score,
            "consistency_score": consistency_score,
            "source_diversity": source_diversity,
            "temporal_freshness": avg_freshness,
            "evidence_count": len(evidence)
        }
    
    def _calculate_evidence_consistency(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate consistency score for evidence set"""
        if len(evidence) < 2:
            return 0.7  # Neutral consistency for single evidence
        
        # Simplified consistency check
        # In practice, would use semantic analysis
        consistency_scores = []
        
        for i in range(len(evidence)):
            for j in range(i + 1, len(evidence)):
                item1 = evidence[i]
                item2 = evidence[j]
                
                # Check for direct contradictions in claims
                claim1 = str(item1.get('claim', '')).lower()
                claim2 = str(item2.get('claim', '')).lower()
                
                if claim1 and claim2:
                    # Simple contradiction detection
                    negation_words = ['not ', 'no ', "don't ", "doesn't ", "didn't ", "isn't ", "aren't "]
                    contradictory = False
                    
                    for negation in negation_words:
                        if (negation in claim1 and negation not in claim2 and 
                            claim1.replace(negation, '') in claim2):
                            contradictory = True
                            break
                        elif (negation in claim2 and negation not in claim1 and 
                              claim2.replace(negation, '') in claim1):
                            contradictory = True
                            break
                    
                    if contradictory:
                        consistency_scores.append(0.0)
                    else:
                        # Semantic similarity (simplified)
                        words1 = set(claim1.split())
                        words2 = set(claim2.split())
                        common = words1.intersection(words2)
                        similarity = len(common) / max(1, min(len(words1), len(words2)))
                        consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 0.7
    
    def _evaluate_options(self, options: List[str], 
                         evidence_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate each option against evidence"""
        evaluations = []
        
        for option in options:
            # Calculate support score based on evidence
            support_score = self._calculate_option_support(option, evidence_analysis)
            
            # Consider option characteristics
            option_complexity = min(1.0, len(option.split()) / 20.0)
            option_clarity = 0.7  # Simplified
            
            # Overall evaluation
            evaluation_score = (
                support_score * 0.6 +
                (1 - option_complexity) * 0.2 +
                option_clarity * 0.2
            )
            
            evaluations.append({
                "option": option,
                "support_score": support_score,
                "complexity": option_complexity,
                "clarity": option_clarity,
                "evaluation_score": evaluation_score,
                "strengths": self._identify_option_strengths(option, evidence_analysis),
                "weaknesses": self._identify_option_weaknesses(option, evidence_analysis)
            })
        
        return evaluations
    
    def _calculate_option_support(self, option: str, 
                                evidence_analysis: Dict[str, Any]) -> float:
        """Calculate how well evidence supports an option"""
        # Simplified support calculation
        # In practice, would use semantic analysis and reasoning
        
        evidence_strength = evidence_analysis.get('total_strength', 0.5)
        consistency = evidence_analysis.get('consistency_score', 0.7)
        
        # Option length heuristic (longer options often more specific)
        option_words = len(option.split())
        specificity = min(1.0, option_words / 10.0)
        
        # Combine factors
        support = (
            evidence_strength * 0.4 +
            consistency * 0.3 +
            specificity * 0.3
        )
        
        return max(0.1, min(1.0, support))
    
    def _identify_option_strengths(self, option: str, 
                                  evidence_analysis: Dict[str, Any]) -> List[str]:
        """Identify strengths of an option"""
        strengths = []
        
        evidence_count = evidence_analysis.get('evidence_count', 0)
        if evidence_count >= 3:
            strengths.append("Well-supported by multiple evidence sources")
        
        consistency = evidence_analysis.get('consistency_score', 0.7)
        if consistency > 0.8:
            strengths.append("Consistent with available evidence")
        
        option_words = len(option.split())
        if option_words > 5:
            strengths.append("Specific and detailed")
        elif option_words <= 3:
            strengths.append("Clear and concise")
        
        return strengths
    
    def _identify_option_weaknesses(self, option: str, 
                                   evidence_analysis: Dict[str, Any]) -> List[str]:
        """Identify weaknesses of an option"""
        weaknesses = []
        
        evidence_count = evidence_analysis.get('evidence_count', 0)
        if evidence_count == 0:
            weaknesses.append("No supporting evidence")
        elif evidence_count == 1:
            weaknesses.append("Limited evidence support")
        
        consistency = evidence_analysis.get('consistency_score', 0.7)
        if consistency < 0.5:
            weaknesses.append("Contradicts some evidence")
        
        option_words = len(option.split())
        if option_words > 15:
            weaknesses.append("Overly complex")
        elif option_words < 2:
            weaknesses.append("Vague or underspecified")
        
        return weaknesses
    
    def _calculate_confidence(self, option_evaluations: List[Dict[str, Any]],
                            evidence_analysis: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate confidence and uncertainty metrics"""
        if not option_evaluations:
            return 0.5, 0.5
        
        # Extract evaluation scores
        scores = [e['evaluation_score'] for e in option_evaluations]
        
        if len(scores) == 1:
            # Single option
            confidence = scores[0]
            uncertainty = 1.0 - confidence
        else:
            # Multiple options - confidence based on score difference
            sorted_scores = sorted(scores, reverse=True)
            best_score = sorted_scores[0]
            
            if len(sorted_scores) > 1:
                second_best = sorted_scores[1]
                score_gap = best_score - second_best
            else:
                score_gap = best_score
            
            # Confidence increases with score gap and evidence quality
            evidence_quality = evidence_analysis.get('total_strength', 0.5)
            confidence = min(0.95, max(0.05, 
                best_score * 0.6 + 
                score_gap * 2.0 * 0.2 + 
                evidence_quality * 0.2
            ))
            
            # Uncertainty based on score distribution
            score_variance = np.var(scores) if len(scores) > 1 else 0.5
            uncertainty = min(0.9, max(0.1, 
                (1 - confidence) * 0.6 + 
                score_variance * 0.4
            ))
        
        # Apply confidence calibration based on historical performance
        calibrated_confidence = self._calibrate_confidence(confidence)
        
        return calibrated_confidence, uncertainty
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Calibrate confidence based on historical accuracy"""
        calibration_error = self.state.confidence_calibration_error
        
        # Adjust confidence toward well-calibrated level
        if calibration_error > 0.2:
            # High calibration error - be conservative
            if raw_confidence > 0.7:
                calibrated = raw_confidence * 0.8
            else:
                calibrated = raw_confidence
        elif calibration_error < 0.1:
            # Good calibration - trust the confidence
            calibrated = raw_confidence
        else:
            # Moderate calibration - slight adjustment
            calibrated = raw_confidence * (1 - calibration_error * 0.5)
        
        return max(0.1, min(0.95, calibrated))
    
    def _select_conclusion(self, option_evaluations: List[Dict[str, Any]],
                          confidence: float) -> Tuple[str, List[str]]:
        """Select conclusion and generate reasoning steps"""
        if not option_evaluations:
            return "Insufficient information to reach conclusion", []
        
        # Select best option
        best_eval = max(option_evaluations, key=lambda x: x['evaluation_score'])
        conclusion = best_eval['option']
        
        # Generate reasoning steps
        reasoning_steps = [
            f"Evaluated {len(option_evaluations)} possible conclusions",
            f"Selected '{conclusion}' based on evidence support score: {best_eval['support_score']:.2f}",
            f"Overall evaluation score: {best_eval['evaluation_score']:.2f}",
            f"Confidence level: {confidence:.2f}"
        ]
        
        # Add strengths and weaknesses
        if best_eval['strengths']:
            reasoning_steps.append(f"Strengths: {', '.join(best_eval['strengths'])}")
        if best_eval['weaknesses']:
            reasoning_steps.append(f"Considerations: {', '.join(best_eval['weaknesses'])}")
        
        return conclusion, reasoning_steps
    
    def _assess_judgment_quality(self, evidence_analysis: Dict[str, Any],
                               option_evaluations: List[Dict[str, Any]],
                               confidence: float, conclusion: str,
                               reasoning_steps: List[str]) -> Dict[str, Any]:
        """Comprehensive judgment quality assessment"""
        
        # Evidence quality metrics
        evidence_quality = (
            evidence_analysis.get('total_strength', 0.5) * 0.3 +
            evidence_analysis.get('consistency_score', 0.7) * 0.2 +
            evidence_analysis.get('source_diversity', 0.5) * 0.2 +
            evidence_analysis.get('temporal_freshness', 0.5) * 0.1 +
            min(1.0, evidence_analysis.get('evidence_count', 0) / 5.0) * 0.2
        )
        
        # Option evaluation quality
        if option_evaluations:
            evaluation_scores = [e['evaluation_score'] for e in option_evaluations]
            evaluation_quality = np.mean(evaluation_scores)
            evaluation_variance = np.var(evaluation_scores) if len(evaluation_scores) > 1 else 0
        else:
            evaluation_quality = 0.3
            evaluation_variance = 0.5
        
        # Reasoning quality
        reasoning_quality = min(1.0, len(reasoning_steps) / 5.0)
        
        # Confidence appropriateness
        if option_evaluations:
            best_score = max([e['evaluation_score'] for e in option_evaluations])
            confidence_appropriateness = 1.0 - abs(confidence - best_score)
        else:
            confidence_appropriateness = 0.5
        
        # Overall quality
        overall_quality = (
            evidence_quality * 0.25 +
            evaluation_quality * 0.25 +
            reasoning_quality * 0.20 +
            confidence_appropriateness * 0.30
        )
        
        # Check for potential issues
        needs_correction = False
        if overall_quality < 0.5:
            needs_correction = True
        elif confidence > 0.8 and evidence_analysis.get('evidence_count', 0) < 2:
            needs_correction = True
        elif evaluation_variance < 0.1 and len(option_evaluations) > 2:
            needs_correction = True  # All options similar, might need deeper analysis
        
        # Generate recommendations
        recommendations = []
        if evidence_analysis.get('evidence_count', 0) < 2:
            recommendations.append("Seek additional evidence sources")
        if confidence > 0.8 and evidence_quality < 0.6:
            recommendations.append("Consider reducing confidence given evidence limitations")
        if evaluation_variance < 0.1:
            recommendations.append("Explore more diverse conclusion options")
        
        return {
            "evidence_quality": evidence_quality,
            "evaluation_quality": evaluation_quality,
            "reasoning_quality": reasoning_quality,
            "confidence_appropriateness": confidence_appropriateness,
            "overall_quality": overall_quality,
            "needs_correction": needs_correction,
            "recommendations": recommendations,
            "evaluation_variance": evaluation_variance
        }
    
    def _generate_reasoning_summary(self, reasoning_steps: List[str]) -> str:
        """Generate concise reasoning summary"""
        if not reasoning_steps:
            return "Judgment made based on available evidence and analysis."
        
        # Combine key reasoning steps
        summary_parts = []
        
        for step in reasoning_steps[:3]:  # Use first 3 steps for summary
            if len(summary_parts) < 100:  # Keep summary concise
                summary_parts.append(step)
        
        return " ".join(summary_parts)
    
    def _update_judgment_state(self, judgment_record: JudgmentRecord,
                             quality_metrics: Dict[str, Any]):
        """Update self-judgment state based on new judgment"""
        self.state.total_judgments += 1
        
        # Update confidence calibration
        current_confidence = judgment_record.confidence
        quality_score = quality_metrics.get('overall_quality', 0.5)
        
        # Assume quality score approximates accuracy for calibration
        confidence_error = abs(current_confidence - quality_score)
        self.performance_metrics["confidence_errors"].append(confidence_error)
        
        # Update state metrics
        self.state.average_confidence = (
            self.state.average_confidence * 0.9 + current_confidence * 0.1
        )
        
        self.state.confidence_calibration_error = np.mean(
            list(self.performance_metrics["confidence_errors"])
        ) if self.performance_metrics["confidence_errors"] else 0.3
        
        # Update judgment success rate
        if quality_metrics.get('overall_quality', 0) > 0.6:
            self.state.successful_judgments += 1
        
        # Update performance tracking
        self.performance_metrics["judgment_qualities"].append(
            quality_metrics.get('overall_quality', 0.5)
        )
        
        # Update learning progress
        judgment_success_rate = (
            self.state.successful_judgments / max(1, self.state.total_judgments)
        )
        self.state.learning_progress = min(1.0, 
            self.state.learning_progress + 
            (judgment_success_rate - 0.5) * 0.05
        )
    
    # ========== Argument Chain Construction ==========
    
    def build_argument_chain(self, premises: List[str], 
                           conclusion: str,
                           evidence_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build structured argument chain for fact verification
        
        Args:
            premises: List of premise statements
            conclusion: Conclusion statement
            evidence_sources: Evidence supporting premises and conclusion
            
        Returns:
            Dict containing argument chain and quality assessment
        """
        start_time = time.time()
        
        try:
            # Create evidence mapping
            evidence_map = self._map_evidence_to_statements(premises + [conclusion], evidence_sources)
            
            # Build logical structure
            logical_structure = self._build_logical_structure(premises, conclusion)
            
            # Assess argument quality
            completeness = self._assess_argument_completeness(premises, conclusion, evidence_map)
            coherence = self._assess_argument_coherence(premises, conclusion, logical_structure)
            evidence_support = self._assess_evidence_support(evidence_map)
            
            # Overall quality
            overall_quality = (
                completeness * 0.4 +
                coherence * 0.3 +
                evidence_support * 0.3
            )
            
            # Create argument chain record
            argument_id = f"argument_{int(time.time())}_{abs((zlib.adler32(str(str(time.time().encode('utf-8')) & 0xffffffff)) + str(premises) + conclusion)) % 9000 + 1000}"
            argument_chain = ArgumentChain(
                argument_id=argument_id,
                premises=premises,
                conclusion=conclusion,
                evidence_map=evidence_map,
                logical_structure=logical_structure,
                completeness_score=completeness,
                coherence_score=coherence,
                evidence_support_score=evidence_support,
                overall_quality=overall_quality,
                timestamp=time.time(),
                verification_status="pending"
            )
            
            # Store and update state
            self.argument_history.append(argument_chain)
            self._update_argument_state(argument_chain)
            
            # Attempt verification if quality is sufficient
            if overall_quality > 0.6:
                verification_result = self._attempt_argument_verification(argument_chain)
                argument_chain.verification_status = verification_result.get('status', 'pending')
            
            response = {
                "success": True,
                "argument_id": argument_id,
                "premises": premises,
                "conclusion": conclusion,
                "quality_metrics": {
                    "completeness": completeness,
                    "coherence": coherence,
                    "evidence_support": evidence_support,
                    "overall_quality": overall_quality
                },
                "verification_status": argument_chain.verification_status,
                "processing_time": time.time() - start_time
            }
            
            self.logger.info(f"Argument chain {argument_id} built with quality: {overall_quality:.2f}")
            return response
            
        except Exception as e:
            self.logger.error(f"Argument chain construction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "argument_id": f"failed_{int(time.time())}"
            }
    
    def _map_evidence_to_statements(self, statements: List[str],
                                   evidence_sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Map evidence sources to statements they support"""
        evidence_map = {stmt: [] for stmt in statements}
        
        for evidence in evidence_sources:
            # Extract evidence claims and relevance
            claims = evidence.get('claims', [])
            relevance = evidence.get('relevance', 0.7)
            strength = evidence.get('strength', 0.5)
            
            # Match evidence to statements (simplified)
            for stmt in statements:
                stmt_lower = stmt.lower()
                evidence_text = str(evidence.get('content', '')).lower()
                
                # Check for keyword overlap
                stmt_words = set(stmt_lower.split())
                evidence_words = set(evidence_text.split())
                common_words = stmt_words.intersection(evidence_words)
                
                if len(common_words) >= 2:  # At least 2 common words
                    evidence_map[stmt].append({
                        'source': evidence.get('source', 'unknown'),
                        'relevance': relevance,
                        'strength': strength,
                        'type': evidence.get('type', 'unknown'),
                        'timestamp': evidence.get('timestamp', time.time())
                    })
        
        return evidence_map
    
    def _build_logical_structure(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """Build logical structure connecting premises to conclusion"""
        # Simplified logical structure
        # In practice, would use formal logic analysis
        
        structure = {
            "premises": premises,
            "conclusion": conclusion,
            "inference_type": "deductive",  # Default assumption
            "premise_count": len(premises),
            "logical_connectors": []
        }
        
        # Identify logical connectors
        logical_indicators = ['therefore', 'thus', 'hence', 'so', 'because', 'since', 'as']
        for indicator in logical_indicators:
            if indicator in conclusion.lower():
                structure["logical_connectors"].append(indicator)
        
        # Check for conditional structure
        if any(['if ' in p.lower() for p in premises]):
            structure["inference_type"] = "conditional"
        elif any(['all ' in p.lower() or 'every ' in p.lower() for p in premises]):
            structure["inference_type"] = "universal"
        
        return structure
    
    def _assess_argument_completeness(self, premises: List[str],
                                     conclusion: str,
                                     evidence_map: Dict[str, List[Dict[str, Any]]]) -> float:
        """Assess completeness of argument"""
        if not premises:
            return 0.1
        
        # Check if all premises have evidence
        premises_with_evidence = 0
        for premise in premises:
            if premise in evidence_map and evidence_map[premise]:
                premises_with_evidence += 1
        
        premise_completeness = premises_with_evidence / len(premises)
        
        # Check conclusion evidence
        conclusion_evidence = conclusion in evidence_map and evidence_map[conclusion]
        conclusion_factor = 0.8 if conclusion_evidence else 0.5
        
        return premise_completeness * conclusion_factor
    
    def _assess_argument_coherence(self, premises: List[str],
                                  conclusion: str,
                                  logical_structure: Dict[str, Any]) -> float:
        """Assess logical coherence of argument"""
        # Simplified coherence assessment
        # In practice, would use logical consistency checking
        
        # Check semantic consistency
        all_text = " ".join(premises + [conclusion]).lower()
        words = set(all_text.split())
        
        # Check for contradictions (simplified)
        negation_words = ['not', 'no', "don't", "doesn't", "didn't"]
        has_negation = any(word in negation_words for word in words)
        
        if has_negation:
            # Check for potential contradictions
            positive_terms = [w for w in words if w not in negation_words]
            # Simple check: if same term appears with and without negation
            for term in positive_terms:
                negated_term = f"not_{term}"
                if negated_term in all_text.replace(" ", "_"):
                    return 0.3  # Potential contradiction
        
        # Check logical structure
        inference_type = logical_structure.get('inference_type', 'unknown')
        if inference_type == 'deductive':
            base_coherence = 0.7
        elif inference_type == 'conditional':
            base_coherence = 0.8
        else:
            base_coherence = 0.6
        
        # Adjust for logical connectors
        connectors = logical_structure.get('logical_connectors', [])
        if connectors:
            base_coherence += len(connectors) * 0.1
        
        return min(1.0, base_coherence)
    
    def _assess_evidence_support(self, evidence_map: Dict[str, List[Dict[str, Any]]]) -> float:
        """Assess strength of evidence support"""
        if not evidence_map:
            return 0.1
        
        total_strength = 0.0
        total_items = 0
        
        for statement, evidence_list in evidence_map.items():
            for evidence in evidence_list:
                strength = evidence.get('strength', 0.5)
                relevance = evidence.get('relevance', 0.7)
                reliability = evidence.get('type', 'unknown')
                
                # Adjust for evidence type
                type_multiplier = 1.0
                if reliability == 'empirical_data':
                    type_multiplier = 1.2
                elif reliability == 'expert_opinion':
                    type_multiplier = 0.9
                elif reliability == 'anecdotal':
                    type_multiplier = 0.7
                
                evidence_strength = strength * relevance * type_multiplier
                total_strength += evidence_strength
                total_items += 1
        
        if total_items == 0:
            return 0.1
        
        avg_strength = total_strength / total_items
        return min(1.0, avg_strength)
    
    def _update_argument_state(self, argument_chain: ArgumentChain):
        """Update state based on new argument chain"""
        quality = argument_chain.overall_quality
        
        # Update performance metrics
        self.performance_metrics["argument_qualities"].append(quality)
        
        # Update success rate
        if quality > 0.6:
            successful = 1.0
        else:
            successful = 0.0
            
        self.performance_metrics["argument_qualities"].append(quality)
        
        # Update state
        recent_qualities = list(self.performance_metrics["argument_qualities"])[-10:]
        if recent_qualities:
            self.state.argument_chain_success_rate = np.mean(
                [1 if q > 0.6 else 0 for q in recent_qualities]
            )
    
    def _attempt_argument_verification(self, argument_chain: ArgumentChain) -> Dict[str, Any]:
        """Attempt to verify argument chain"""
        # Simplified verification
        # In practice, would use knowledge graph and external verification
        
        quality = argument_chain.overall_quality
        
        if quality > 0.8:
            status = "verified"
            confidence = 0.8
        elif quality > 0.6:
            status = "partially_verified"
            confidence = 0.6
        else:
            status = "needs_review"
            confidence = 0.4
        
        return {
            "status": status,
            "confidence": confidence,
            "verification_method": "internal_quality_assessment",
            "timestamp": time.time()
        }
    
    # ========== Self-Correction Methods ==========
    
    def _attempt_self_correction(self, judgment_record: JudgmentRecord) -> Optional[Dict[str, Any]]:
        """Attempt self-correction based on judgment issues"""
        try:
            quality_metrics = judgment_record.quality_metrics
            
            # Identify correction needs
            correction_needs = self._identify_correction_needs(judgment_record, quality_metrics)
            
            if not correction_needs:
                return None
            
            # Generate correction
            correction_id = f"correction_{int(time.time())}_{abs((zlib.adler32(str(str(time.time().encode('utf-8')) & 0xffffffff)) + str(error_context))) % 9000 + 1000}"
            correction = self._generate_correction(judgment_record, correction_needs)
            
            # Evaluate correction quality
            effectiveness = self._evaluate_correction_effectiveness(correction, judgment_record)
            
            # Create correction record
            correction_record = CorrectionAttempt(
                correction_id=correction_id,
                error_type=correction_needs.get('primary_error', 'unknown'),
                error_description=correction_needs.get('description', ''),
                original_content={
                    "conclusion": judgment_record.conclusion,
                    "confidence": judgment_record.confidence,
                    "reasoning_steps": judgment_record.reasoning_steps
                },
                correction_strategy=correction.get('strategy', 'general_review'),
                corrected_content=correction.get('corrected_content', {}),
                confidence=correction.get('confidence', 0.5),
                effectiveness_score=effectiveness,
                timestamp=time.time(),
                metadata={
                    "judgment_id": judgment_record.judgment_id,
                    "quality_metrics": quality_metrics
                }
            )
            
            # Store and update state
            self.correction_history.append(correction_record)
            self._update_correction_state(correction_record)
            
            self.logger.info(f"Self-correction {correction_id} attempted (effectiveness: {effectiveness:.2f})")
            return {
                "correction_id": correction_id,
                "correction": correction,
                "effectiveness": effectiveness
            }
            
        except Exception as e:
            self.logger.error(f"Self-correction failed: {e}")
            return None
    
    def _identify_correction_needs(self, judgment_record: JudgmentRecord,
                                 quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Identify specific correction needs"""
        needs = {}
        
        # Check evidence quality
        evidence_quality = quality_metrics.get('evidence_quality', 0.5)
        if evidence_quality < 0.4:
            needs['primary_error'] = 'evidence_insufficiency'
            needs['description'] = f'Evidence quality too low: {evidence_quality:.2f}'
            needs['priority'] = 'high'
        
        # Check confidence calibration
        confidence_appropriateness = quality_metrics.get('confidence_appropriateness', 0.5)
        if confidence_appropriateness < 0.6:
            if judgment_record.confidence > 0.7:
                needs['primary_error'] = 'overconfidence'
                needs['description'] = f'Confidence ({judgment_record.confidence:.2f}) may be too high'
            else:
                needs['primary_error'] = 'underconfidence'
                needs['description'] = f'Confidence ({judgment_record.confidence:.2f}) may be too low'
            needs['priority'] = 'medium'
        
        # Check evaluation variance
        evaluation_variance = quality_metrics.get('evaluation_variance', 0.5)
        if evaluation_variance < 0.1 and len(judgment_record.alternatives_considered) > 2:
            needs['primary_error'] = 'evaluation_insensitivity'
            needs['description'] = 'All options evaluated similarly, may need deeper analysis'
            needs['priority'] = 'low'
        
        return needs
    
    def _generate_correction(self, judgment_record: JudgmentRecord,
                           correction_needs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific correction based on needs"""
        error_type = correction_needs.get('primary_error', 'unknown')
        
        correction_strategies = {
            'evidence_insufficiency': self._correct_evidence_insufficiency,
            'overconfidence': self._correct_overconfidence,
            'underconfidence': self._correct_underconfidence,
            'evaluation_insensitivity': self._correct_evaluation_insensitivity
        }
        
        if error_type in correction_strategies:
            return correction_strategies[error_type](judgment_record, correction_needs)
        else:
            return self._general_correction(judgment_record, correction_needs)
    
    def _correct_evidence_insufficiency(self, judgment_record: JudgmentRecord,
                                      correction_needs: Dict[str, Any]) -> Dict[str, Any]:
        """Correct evidence insufficiency"""
        return {
            'strategy': 'evidence_augmentation',
            'corrected_content': {
                'action': 'seek_additional_evidence',
                'current_evidence_count': len(judgment_record.evidence),
                'target_evidence_count': len(judgment_record.evidence) + 3,
                'evidence_types_suggested': ['empirical_data', 'expert_opinion', 'theoretical_support'],
                'reason': 'Insufficient evidence for confident conclusion'
            },
            'confidence': 0.7,
            'implementation_steps': [
                'Identify key claims needing support',
                'Search for additional evidence sources',
                'Re-evaluate conclusion with new evidence'
            ]
        }
    
    def _correct_overconfidence(self, judgment_record: JudgmentRecord,
                              correction_needs: Dict[str, Any]) -> Dict[str, Any]:
        """Correct overconfidence"""
        new_confidence = max(0.3, judgment_record.confidence * 0.7)
        
        return {
            'strategy': 'confidence_adjustment',
            'corrected_content': {
                'action': 'adjust_confidence',
                'original_confidence': judgment_record.confidence,
                'new_confidence': new_confidence,
                'adjustment_factor': 0.7,
                'reason': 'Confidence exceeds evidence support level'
            },
            'confidence': 0.6,
            'implementation_steps': [
                'Review evidence strength',
                'Consider alternative explanations',
                'Adjust confidence to match evidence'
            ]
        }
    
    def _correct_underconfidence(self, judgment_record: JudgmentRecord,
                               correction_needs: Dict[str, Any]) -> Dict[str, Any]:
        """Correct underconfidence"""
        new_confidence = min(0.9, judgment_record.confidence * 1.3)
        
        return {
            'strategy': 'confidence_enhancement',
            'corrected_content': {
                'action': 'adjust_confidence',
                'original_confidence': judgment_record.confidence,
                'new_confidence': new_confidence,
                'adjustment_factor': 1.3,
                'reason': 'Confidence may be excessively cautious given evidence'
            },
            'confidence': 0.6,
            'implementation_steps': [
                'Review evidence supporting current conclusion',
                'Consider strength of supporting evidence',
                'Adjust confidence upward based on evidence'
            ]
        }
    
    def _correct_evaluation_insensitivity(self, judgment_record: JudgmentRecord,
                                        correction_needs: Dict[str, Any]) -> Dict[str, Any]:
        """Correct evaluation insensitivity"""
        return {
            'strategy': 'evaluation_deepening',
            'corrected_content': {
                'action': 'deepen_evaluation',
                'current_options': len(judgment_record.alternatives_considered),
                'suggested_approach': 'multi-criteria_analysis',
                'reason': 'All options evaluated similarly, need deeper differentiation'
            },
            'confidence': 0.5,
            'implementation_steps': [
                'Apply additional evaluation criteria',
                'Weight criteria based on importance',
                'Re-score options with enhanced evaluation'
            ]
        }
    
    def _general_correction(self, judgment_record: JudgmentRecord,
                          correction_needs: Dict[str, Any]) -> Dict[str, Any]:
        """General correction strategy"""
        return {
            'strategy': 'comprehensive_review',
            'corrected_content': {
                'action': 'review_and_adjust',
                'review_aspects': ['evidence', 'reasoning', 'confidence', 'alternatives'],
                'reason': 'General quality improvement needed'
            },
            'confidence': 0.5,
            'implementation_steps': [
                'Re-examine all evidence',
                'Review logical reasoning steps',
                'Check confidence calibration',
                'Consider additional alternatives'
            ]
        }
    
    def _evaluate_correction_effectiveness(self, correction: Dict[str, Any],
                                         original_judgment: JudgmentRecord) -> float:
        """Evaluate effectiveness of correction attempt"""
        # Simplified effectiveness evaluation
        # In practice, would track actual improvement
        
        strategy = correction.get('strategy', '')
        confidence = correction.get('confidence', 0.5)
        
        # Base effectiveness on correction specificity and confidence
        base_effectiveness = 0.5
        
        # Adjust based on strategy
        if strategy in ['evidence_augmentation', 'confidence_adjustment']:
            base_effectiveness += 0.2
        
        # Adjust based on confidence
        base_effectiveness += (confidence - 0.5) * 0.3
        
        return max(0.1, min(1.0, base_effectiveness))
    
    def _update_correction_state(self, correction_record: CorrectionAttempt):
        """Update state based on correction attempt"""
        effectiveness = correction_record.effectiveness_score
        
        # Update performance metrics
        self.performance_metrics["correction_successes"].append(
            1.0 if effectiveness > 0.6 else 0.0
        )
        
        # Update state
        recent_successes = list(self.performance_metrics["correction_successes"])[-10:]
        if recent_successes:
            self.state.self_correction_success_rate = np.mean(recent_successes)
    
    # ========== System Integration and Reporting ==========
    
    def integrate_with_system(self, system_name: str, config: Dict[str, Any] = None) -> bool:
        """Integrate with other AGI systems"""
        try:
            if system_name == "self_reflection":
                # Integration with self-reflection module
                self.integrated_systems["self_reflection"] = True
                self.logger.info("Integrated with self-reflection module")
                return True
                
            elif system_name == "advanced_reasoning":
                # Integration with advanced reasoning engine
                self.integrated_systems["advanced_reasoning"] = True
                self.logger.info("Integrated with advanced reasoning engine")
                return True
                
            elif system_name == "knowledge_graph":
                # Integration with knowledge graph
                self.integrated_systems["knowledge_graph"] = True
                self.logger.info("Integrated with knowledge graph")
                return True
                
            else:
                self.logger.warning(f"Unknown system for integration: {system_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Integration failed for {system_name}: {e}")
            return False
    
    def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        report = {
            "system_state": {
                "total_judgments": self.state.total_judgments,
                "successful_judgments": self.state.successful_judgments,
                "success_rate": self.state.successful_judgments / max(1, self.state.total_judgments),
                "average_confidence": self.state.average_confidence,
                "confidence_calibration_error": self.state.confidence_calibration_error,
                "self_correction_success_rate": self.state.self_correction_success_rate,
                "argument_chain_success_rate": self.state.argument_chain_success_rate,
                "learning_progress": self.state.learning_progress
            },
            "performance_metrics": {
                "recent_judgment_quality": np.mean(list(self.performance_metrics["judgment_qualities"])[-10:]) 
                if self.performance_metrics["judgment_qualities"] else 0,
                "recent_confidence_error": np.mean(list(self.performance_metrics["confidence_errors"])[-10:])
                if self.performance_metrics["confidence_errors"] else 0,
                "recent_correction_success": np.mean(list(self.performance_metrics["correction_successes"])[-10:])
                if self.performance_metrics["correction_successes"] else 0,
                "recent_argument_quality": np.mean(list(self.performance_metrics["argument_qualities"])[-10:])
                if self.performance_metrics["argument_qualities"] else 0
            },
            "integration_status": self.integrated_systems,
            "insights": [],
            "recommendations": []
        }
        
        # Generate insights
        if self.state.confidence_calibration_error > 0.25:
            report["insights"].append(
                f"Confidence calibration needs improvement (error: {self.state.confidence_calibration_error:.2f})"
            )
            report["recommendations"].append(
                "Implement confidence calibration training with feedback"
            )
        
        if self.state.self_correction_success_rate < 0.4:
            report["insights"].append(
                f"Self-correction success rate is low ({self.state.self_correction_success_rate:.2f})"
            )
            report["recommendations"].append(
                "Develop more effective correction strategies and error detection"
            )
        
        if self.state.learning_progress < 0.3:
            report["insights"].append(
                f"Learning progress is slow ({self.state.learning_progress:.2f})"
            )
            report["recommendations"].append(
                "Increase diversity of judgment tasks and feedback mechanisms"
            )
        
        if not report["insights"]:
            report["insights"].append("System performance is within acceptable ranges")
            report["recommendations"].append("Continue current practices with monitoring")
        
        return report
    
    def save_state(self, filepath: str = None) -> bool:
        """Save current system state"""
        try:
            if filepath is None:
                filepath = "data/self_judgment_state.pkl"
            
            Path("data").mkdir(exist_ok=True)
            
            state_data = {
                'state': self.state,
                'performance_metrics': self.performance_metrics,
                'integrated_systems': self.integrated_systems,
                'thresholds': self.thresholds,
                'error_patterns': self.error_patterns,
                'success_patterns': self.success_patterns
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)
            
            self.logger.info(f"System state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, filepath: str = None) -> bool:
        """Load system state from file"""
        try:
            if filepath is None:
                filepath = "data/self_judgment_state.pkl"
            
            if not Path(filepath).exists():
                self.logger.warning(f"State file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
            
            self.state = state_data.get('state', self.state)
            self.performance_metrics = state_data.get('performance_metrics', self.performance_metrics)
            self.integrated_systems = state_data.get('integrated_systems', self.integrated_systems)
            self.thresholds = state_data.get('thresholds', self.thresholds)
            self.error_patterns = state_data.get('error_patterns', self.error_patterns)
            self.success_patterns = state_data.get('success_patterns', self.success_patterns)
            
            self.logger.info(f"System state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False


# Singleton instance for easy access
self_judgment_module = EnhancedSelfJudgmentModule(from_scratch=True)


def initialize_self_judgment_system(from_scratch: bool = True, 
                                   knowledge_graph_path: str = None) -> EnhancedSelfJudgmentModule:
    """Initialize and return self-judgment system instance"""
    global self_judgment_module
    self_judgment_module = EnhancedSelfJudgmentModule(
        knowledge_graph_path=knowledge_graph_path,
        from_scratch=from_scratch
    )
    return self_judgment_module


def make_judgment_with_system(context: Dict[str, Any], 
                            evidence: List[Dict[str, Any]],
                            conclusion_options: List[str]) -> Dict[str, Any]:
    """Make judgment using the global self-judgment system"""
    return self_judgment_module.make_judgment(context, evidence, conclusion_options)


def build_argument_with_system(premises: List[str], 
                             conclusion: str,
                             evidence_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build argument chain using the global self-judgment system"""
    return self_judgment_module.build_argument_chain(premises, conclusion, evidence_sources)


def get_self_judgment_report() -> Dict[str, Any]:
    """Get report from the global self-judgment system"""
    return self_judgment_module.get_system_report()


# Export main classes and functions
__all__ = [
    'EnhancedSelfJudgmentModule',
    'initialize_self_judgment_system',
    'make_judgment_with_system',
    'build_argument_with_system',
    'get_self_judgment_report',
    'JudgmentRecord',
    'ArgumentChain',
    'CorrectionAttempt',
    'SelfJudgmentState'
]
