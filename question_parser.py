"""
Question Parser - Fixed and Optimized
Correctly parses all question types with improved pattern matching
"""

import re
from typing import Dict, Any

class QuestionParser:
    def __init__(self):
        """Initialize with comprehensive patterns."""
        # Comprehensive patterns for all question types
        self.patterns = {
            # Intent patterns
            'count': r'how many|count|number|total|quantity|gimme.*count|what.*count|how much',
            'average': r'average|mean|avg|typical|normal|what.*is.*the.*average|what.*is.*the.*mean',
            'max': r'highest|maximum|max|top|peak|what.*highest|what.*is.*the.*highest|what.*is.*the.*maximum',
            'min': r'lowest|minimum|min|bottom|least|what.*lowest|what.*is.*the.*lowest|what.*is.*the.*minimum',
            'show': r'show|list|which|what.*show|display|find.*all|what.*is|what.*are|tell.*me',
            'top': r'top \d+|top \d+.*with|highest \d+|maximum \d+|top.*\d+',
            
            # Equipment patterns
            'pump': r'pump|pumps|bomba',
            'motor': r'motor|motors|engine',
            'valve': r'valve|valves|válvula',
            'compressor': r'compressor|comp|compresor',
            'vessel': r'vessel|tank|container',
            'transformer': r'transformer|trans',
            
            # Metric patterns (with typos)
            'temperature': r'temperature|temp|heat|tempreture|temprature',
            'vibration': r'vibration|vibrat|vibrat.*level|vibrat.*leve',
            'pressure': r'pressure|press|presure|pres',
            'runtime': r'runtime|hours|time|operating.*time',
            
            # Condition patterns
            'outdated': r'outdated|obsolete|old|antiquated|desactualizado',
            'high_risk': r'high risk|risky|dangerous|critical|urgent|high.*risk',
            'broken': r'broken|faulty|failed|defective'
        }
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse question with comprehensive understanding."""
        q = question.lower().strip()
        
        # Clean question
        q = re.sub(r'[^\w\s]', ' ', q)
        q = re.sub(r'\s+', ' ', q)
        
        # Detect components
        intent = self._detect_intent(q)
        equipment = self._detect_equipment(q)
        metric = self._detect_metric(q)
        condition = self._detect_condition(q)
        
        # Smart intent detection for ambiguous questions
        if intent == 'show' and metric and not equipment and not condition:
            # "What is the temperature?" -> average
            intent = 'average'
        elif intent == 'show' and condition:
            # "Which machines are outdated?" -> show with condition
            intent = 'show'
        elif intent == 'show' and equipment:
            # "Show me pumps" -> show equipment
            intent = 'show'
        
        # Generate analysis parameters
        if intent == 'count':
            return {
                "success": True,
                "analysis_type": "count",
                "parameters": {"equipment_type": equipment},
                "original_question": question
            }
        elif intent == 'average':
            return {
                "success": True,
                "analysis_type": "average",
                "parameters": {"column": metric},
                "original_question": question
            }
        elif intent == 'max':
            return {
                "success": True,
                "analysis_type": "max",
                "parameters": {"column": metric},
                "original_question": question
            }
        elif intent == 'min':
            return {
                "success": True,
                "analysis_type": "min",
                "parameters": {"column": metric},
                "original_question": question
            }
        elif intent == 'top':
            # Extract number from "top N" pattern
            top_match = re.search(r'top (\d+)', q)
            top_count = int(top_match.group(1)) if top_match else 5
            return {
                "success": True,
                "analysis_type": "top",
                "parameters": {"column": metric, "count": top_count},
                "original_question": question
            }
        elif intent == 'show':
            if condition == 'outdated':
                return {
                    "success": True,
                    "analysis_type": "show_outdated",
                    "parameters": {},
                    "original_question": question
                }
            elif condition == 'high_risk':
                return {
                    "success": True,
                    "analysis_type": "show_high_risk",
                    "parameters": {},
                    "original_question": question
                }
            elif equipment:
                return {
                    "success": True,
                    "analysis_type": "show_equipment",
                    "parameters": {"equipment_type": equipment},
                    "original_question": question
                }
            else:
                return {
                    "success": True,
                    "analysis_type": "show_all",
                    "parameters": {},
                    "original_question": question
                }
        else:
            return {
                "success": True,
                "analysis_type": "show_all",
                "parameters": {},
                "original_question": question
            }
    
    def _detect_intent(self, q: str) -> str:
        """Detect question intent with priority order."""
        # Check for specific intents first (top, count, average, max, min, show)
        intent_priority = ['top', 'count', 'average', 'max', 'min', 'show']
        
        for intent in intent_priority:
            if intent in self.patterns and re.search(self.patterns[intent], q):
                return intent
        
        return 'show'  # Default fallback
    
    def _detect_equipment(self, q: str) -> str:
        """Detect equipment type."""
        equipment_types = ['pump', 'motor', 'valve', 'compressor', 'vessel', 'transformer']
        
        for equipment in equipment_types:
            if equipment in self.patterns and re.search(self.patterns[equipment], q):
                return equipment.title()
        
        return None
    
    def _detect_metric(self, q: str) -> str:
        """Detect metric with column mapping."""
        metric_map = {
            'temperature': 'temperature',
            'vibration': 'vibration_level', 
            'pressure': 'pressure',
            'runtime': 'runtime_hours'
        }
        
        for metric in metric_map.keys():
            if metric in self.patterns and re.search(self.patterns[metric], q):
                return metric_map[metric]
        
        return None
    
    def _detect_condition(self, q: str) -> str:
        """Detect condition."""
        conditions = ['outdated', 'high_risk', 'broken']
        
        for condition in conditions:
            if condition in self.patterns and re.search(self.patterns[condition], q):
                return condition
        
        return None