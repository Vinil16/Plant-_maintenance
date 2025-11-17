import re
from typing import Dict, Any

class QuestionParser:
    """Parses natural language questions into structured queries."""
    
    def __init__(self):
        # Regex patterns for detecting intents, equipment types, and metrics
        self.patterns = {
            'count': r'how many|count|number|total|quantity|gimme.*count|what.*count|how much',
            'average': r'average|mean|avg|typical|normal|what.*is.*the.*average|what.*is.*the.*mean',
            'max': r'highest|maximum|max|top|peak|what.*highest|what.*is.*the.*highest|what.*is.*the.*maximum',
            'min': r'lowest|minimum|min|bottom|least|what.*lowest|what.*is.*the.*lowest|what.*is.*the.*minimum',
            'show': r'show|list|which|what.*show|display|find.*all|what.*is|what.*are|tell.*me',
            'top': r'top \d+|top \d+.*with|highest \d+|maximum \d+|top.*\d+',
            
            'pump': r'pump|pumps|bomba',
            'motor': r'motor|motors|engine',
            'valve': r'valve|valves|válvula',
            'compressor': r'compressor|comp|compresor',
            'vessel': r'vessel|tank|container',
            'transformer': r'transformer|trans',
            
            'temperature': r'temperature|temp|heat|tempreture|temprature',
            'vibration': r'vibration|vibrat|vibrat.*level|vibrat.*leve',
            'pressure': r'pressure|press|presure|pres',
            'runtime': r'runtime|hours|time|operating.*time',
            
            'outdated': r'outdated|obsolete|old|antiquated|desactualizado',
            'high_risk': r'high risk|risky|dangerous|critical|urgent|high.*risk',
            'preventive': r'preventive|prevent|maintenance.*required|need.*maintenance|require.*maintenance',
            'broken': r'broken|faulty|failed|defective',
            
            'predict': r'predict|forecast|prediction|will.*fail|likely.*fail|failure.*prediction',
            'maintenance': r'maintenance|maintain|schedule.*maintenance|when.*maintain|maintenance.*schedule',
            'risk': r'risk|risk.*level|risk.*assessment|assess.*risk|risk.*analysis',
            'failure': r'failure|fail|breakdown|malfunction|defect'
        }
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        # Normalize question: lowercase, remove special chars, collapse whitespace
        q = question.lower().strip()
        q = re.sub(r'[^\w\s]', ' ', q)
        q = re.sub(r'\s+', ' ', q)
        
        # Detect question components
        intent = self._detect_intent(q)
        equipment = self._detect_equipment(q)
        metric = self._detect_metric(q)
        condition = self._detect_condition(q)
        
        # Handle "top N" queries with risk levels first, before predictive parser
        if intent == 'top':
            top_match = re.search(r'top (\d+)', q)
            if top_match:
                if 'low risk' in q or 'low-risk' in q:
                    return {
                        "success": True,
                        "analysis_type": "top_low_risk",
                        "parameters": {"count": int(top_match.group(1))},
                        "original_question": question
                    }
                elif self._has_pattern(q, 'high_risk') or self._has_pattern(q, 'preventive') or 'high risk' in q or 'preventive' in q:
                    return {
                        "success": True,
                        "analysis_type": "top_high_risk",
                        "parameters": {"count": int(top_match.group(1))},
                        "original_question": question
                    }
        
        # Only go to predictive parser if we actually detect relevant keywords
        # Don't process very short unclear questions (like "hi") through predictive parser
        has_predictive_keywords = (self._has_pattern(q, 'predict') or self._has_pattern(q, 'maintenance') or 
                                   self._has_pattern(q, 'risk') or self._has_pattern(q, 'high_risk') or
                                   'failure probability' in q or 'probability' in q or
                                   'fail' in q or 'will' in q)
        
        if has_predictive_keywords and len(q.split()) >= 2:
            return self._parse_predictive_question(q, question)
        
        if intent == 'show' and metric and not equipment and not condition:
            intent = 'average'
        elif intent == 'show' and condition:
            intent = 'show'
        elif intent == 'show' and equipment:
            intent = 'show'
        
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
            top_match = re.search(r'top (\d+)', q)
            top_count = int(top_match.group(1)) if top_match else 5
            
            # Check for risk level in the query
            if self._has_pattern(q, 'high_risk') or self._has_pattern(q, 'preventive') or 'high risk' in q or 'preventive' in q:
                return {
                    "success": True,
                    "analysis_type": "top_high_risk",
                    "parameters": {"count": top_count},
                    "original_question": question
                }
            elif 'low risk' in q or 'low-risk' in q:
                return {
                    "success": True,
                    "analysis_type": "top_low_risk",
                    "parameters": {"count": top_count},
                    "original_question": question
                }
            else:
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
                # If "show" intent but no specific thing to show, ask for clarification
                return {
                    "success": False,
                    "error": "I couldn't understand what you want to see. Please be more specific, for example: 'Show me all pumps' or 'Show high-risk equipment'.",
                    "original_question": question
                }
        elif intent is None:
            # No intent detected - ask user to rephrase
            return {
                "success": False,
                "error": "I couldn't understand your question. Please try rephrasing it or use one of the example questions from the sidebar.",
                "original_question": question
            }
        else:
            # If we can't understand the question, ask user to rephrase
            return {
                "success": False,
                "error": "I couldn't understand your question. Please try rephrasing it or use one of the example questions.",
                "original_question": question
            }
    
    def _detect_intent(self, q: str) -> str | None:
        intent_priority = ['top', 'count', 'average', 'max', 'min', 'show']
        for intent in intent_priority:
            if intent in self.patterns and re.search(self.patterns[intent], q):
                return intent
        # Don't default to 'show' - return None if no intent found
        return None
    
    def _detect_equipment(self, q: str) -> str:
        equipment_types = ['pump', 'motor', 'valve', 'compressor', 'vessel', 'transformer']
        for equipment in equipment_types:
            if equipment in self.patterns and re.search(self.patterns[equipment], q):
                return equipment.title()
        return None
    
    def _detect_metric(self, q: str) -> str:
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
        conditions = ['outdated', 'high_risk', 'broken']
        for condition in conditions:
            if condition in self.patterns and re.search(self.patterns[condition], q):
                return condition
        return None
    
    def _parse_predictive_question(self, q: str, original_question: str) -> Dict[str, Any]:
        top_match = re.search(r'top (\d+)', q.lower())
        top_limit = int(top_match.group(1)) if top_match else None
        
        asset_id = None
        if not top_limit:
            asset_id = self._extract_asset_id(original_question)
        
        nq = q.lower()
        
        if asset_id:
            if 'failure probability' in nq or 'probability' in nq or 'will' in nq and 'fail' in nq or 'predict' in nq or 'failure' in nq:
                return {
                    "success": True,
                    "analysis_type": "predict_failure",
                    "parameters": {"asset_id": asset_id},
                    "original_question": original_question
                }
            if 'risk level' in nq or 'high risk' in nq or 'risk' in nq or 'is' in nq and 'risk' in nq:
                return {
                    "success": True,
                    "analysis_type": "risk_assessment",
                    "parameters": {"asset_id": asset_id},
                    "original_question": original_question
                }
            if 'maintenance' in nq or 'maintain' in nq or 'service' in nq:
                return {
                    "success": True,
                    "analysis_type": "maintenance_schedule",
                    "parameters": {"asset_id": asset_id},
                    "original_question": original_question
                }
        
        if top_limit:
            params = {"limit": top_limit}
            if asset_id:
                params["asset_id"] = asset_id
            return {
                "success": True,
                "analysis_type": "predict_failure",
                "parameters": params,
                "original_question": original_question
            }
        
        if self._has_pattern(q, 'high_risk') or self._has_pattern(q, 'preventive') or self._has_pattern(q, 'risky') or 'high risk' in q or 'preventive' in q:
            return {
                "success": True,
                "analysis_type": "show_high_risk",
                "parameters": {},
                "original_question": original_question
            }
        elif self._has_pattern(q, 'maintenance') or self._has_pattern(q, 'maintain'):
            return {
                "success": True,
                "analysis_type": "maintenance_schedule",
                "parameters": {"asset_id": asset_id},
                "original_question": original_question
            }
        elif self._has_pattern(q, 'risk') and asset_id:
            return {
                "success": True,
                "analysis_type": "risk_assessment",
                "parameters": {"asset_id": asset_id},
                "original_question": original_question
            }
        elif self._has_pattern(q, 'predict') or self._has_pattern(q, 'failure'):
            params = {}
            if asset_id:
                params["asset_id"] = asset_id
            return {
                "success": True,
                "analysis_type": "predict_failure",
                "parameters": params,
                "original_question": original_question
            }
        else:
            # If we can't understand the question, ask user to rephrase
            return {
                "success": False,
                "error": "I couldn't understand your question. Please try rephrasing it or use one of the example questions.",
                "original_question": original_question
            }
    
    def _extract_asset_id(self, q: str) -> str:
        text = q.upper()
        
        top_match = re.search(r'\btop\s+(\d+)\b', text)
        if top_match:
            text = re.sub(r'\btop\s+\d+\b', '', text)
        
        exclude_prefixes = {'TOP', 'SHOW', 'LIST', 'COUNT', 'AVG', 'MAX', 'MIN'}
        
        m = re.search(r'\b([A-Z]+)[- ]?(\d{1,4})\b', text)
        if not m:
            return None
        prefix, num = m.group(1), m.group(2)
        
        if prefix in exclude_prefixes:
            return None
        
        try:
            normalized = f"{prefix}-{int(num):03d}"
            return normalized
        except Exception:
            return f"{prefix}-{num}"
    
    def _has_pattern(self, q: str, pattern_name: str) -> bool:
        if pattern_name in self.patterns:
            return bool(re.search(self.patterns[pattern_name], q))
        return False
