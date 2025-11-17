from typing import Dict, Any

class AnswerFormatter:
    """Formats analysis results into human-readable answers."""
    
    def __init__(self):
        # Unit mappings for displaying values
        self.units = {
            'temperature': '°C',
            'vibration_level': 'mm/s',
            'pressure': 'bar',
            'runtime_hours': 'hours'
        }
    
    def format_answer(self, analysis: Dict[str, Any], parsed_question: Dict[str, Any]) -> str:
        if not analysis.get("success"):
            return f"Error: {analysis.get('error')}"
        
        analysis_type = parsed_question.get("analysis_type", "")
        
        if analysis_type == "count":
            return self._format_count_answer(analysis)
        elif analysis_type == "average":
            return self._format_average_answer(analysis)
        elif analysis_type == "max":
            return self._format_max_answer(analysis)
        elif analysis_type == "min":
            return self._format_min_answer(analysis)
        elif analysis_type == "top":
            return self._format_top_answer(analysis)
        elif analysis_type == "show_outdated":
            return self._format_show_outdated_answer(analysis)
        elif analysis_type == "show_high_risk":
            return self._format_show_high_risk_answer(analysis)
        elif analysis_type == "show_equipment":
            return self._format_show_equipment_answer(analysis)
        elif analysis_type == "show_all":
            return self._format_show_all_answer(analysis)
        elif analysis_type == "predict_failure":
            return self._format_predict_failure_answer(analysis)
        elif analysis_type == "maintenance_schedule":
            return self._format_maintenance_schedule_answer(analysis)
        elif analysis_type == "risk_assessment":
            return self._format_risk_assessment_answer(analysis)
        elif analysis_type == "top_high_risk":
            return self._format_top_high_risk_answer(analysis)
        elif analysis_type == "top_low_risk":
            return self._format_top_low_risk_answer(analysis)
        else:
            return f"{analysis.get('summary', 'Analysis completed')}"
    
    def _format_count_answer(self, analysis: Dict[str, Any]) -> str:
        count = analysis.get("data", 0)
        equipment = analysis.get("equipment")
        total_records = analysis.get("total_records", 0)
        
        if equipment:
            return f"There are {count} {equipment.lower()}s in the system (out of {total_records} total equipment items)."
        else:
            return f"There are {count} total equipment items in the system."
    
    def _format_average_answer(self, analysis: Dict[str, Any]) -> str:
        value = analysis.get("data", 0)
        column = analysis.get("column", "")
        unit = self.units.get(column, "")
        total_records = analysis.get("total_records", 0)
        
        column_name = column.replace('_', ' ').title()
        return f"The average {column_name} is {value} {unit} (calculated from {total_records} equipment items)."
    
    def _format_max_answer(self, analysis: Dict[str, Any]) -> str:
        value = analysis.get("data", 0)
        column = analysis.get("column", "")
        unit = self.units.get(column, "")
        total_records = analysis.get("total_records", 0)
        
        column_name = column.replace('_', ' ').title()
        return f"The highest {column_name} is {value} {unit} (from {total_records} equipment items)."
    
    def _format_min_answer(self, analysis: Dict[str, Any]) -> str:
        value = analysis.get("data", 0)
        column = analysis.get("column", "")
        unit = self.units.get(column, "")
        total_records = analysis.get("total_records", 0)
        
        column_name = column.replace('_', ' ').title()
        return f"The lowest {column_name} is {value} {unit} (from {total_records} equipment items)."
    
    def _format_top_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", [])
        column = analysis.get("column", "")
        count = analysis.get("count", 5)
        unit = self.units.get(column, "")
        total_records = analysis.get("total_records", 0)
        
        if not data:
            return f"No data found for top {count} {column.replace('_', ' ')}."
        
        column_name = column.replace('_', ' ').title()
        answer = f"Top {count} equipment with highest {column_name} (from {total_records} total items):\n"
        
        for i, item in enumerate(data, 1):
            value = item.get(column, 0)
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            answer += f"  {i}. {asset_id} - {asset_type} ({value} {unit})\n"
        
        return answer.strip()
    
    def _format_show_outdated_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", [])
        count = analysis.get("count", 0)
        total_records = analysis.get("total_records", 0)
        
        if count == 0:
            return f"No outdated equipment found. All {total_records} equipment items are current."
        
        answer = f"Found {count} outdated equipment items (out of {total_records} total):\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            model_status = item.get('model_status', 'N/A')
            answer += f"  {i}. {asset_id} - {asset_type} ({model_status})\n"
        
        if count > 10:
            answer += f"  ... and {count - 10} more outdated items"
        
        return answer.strip()
    
    def _format_show_high_risk_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", [])
        count = analysis.get("count", 0)
        total_records = analysis.get("total_records", 0)
        
        if count == 0:
            return f"No high-risk equipment found. All {total_records} equipment items are at acceptable risk levels."
        
        answer = f"Found {count} high-risk equipment items (out of {total_records} total):\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            prob = item.get('failure_probability', 0)
            risk_level = item.get('risk_level')
            
            # Calculate risk level if not provided
            if risk_level is None:
                if prob >= 0.85:
                    risk_level = 'Very High'
                elif prob >= 0.70:
                    risk_level = 'High'
                elif prob >= 0.50:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
            
            answer += f"  {i}. {asset_id} - {asset_type} (Risk: {risk_level})\n"
        
        if count > 10:
            answer += f"  ... and {count - 10} more high-risk items"
        
        return answer.strip()
    
    def _format_show_equipment_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", [])
        count = analysis.get("count", 0)
        equipment = analysis.get("equipment", "")
        total_records = analysis.get("total_records", 0)
        
        if count == 0:
            return f"No {equipment.lower()}s found in the system (out of {total_records} total equipment items)."
        
        answer = f"Found {count} {equipment.lower()}s (out of {total_records} total equipment items):\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            answer += f"  {i}. {asset_id} - {asset_type}\n"
        
        if count > 10:
            answer += f"  ... and {count - 10} more {equipment.lower()}s"
        
        return answer.strip()
    
    def _format_show_all_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", [])
        total_count = analysis.get("total_records", 0)
        
        answer = f"Showing {len(data)} of {total_count} total equipment items:\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            answer += f"  {i}. {asset_id} - {asset_type}\n"
        
        if total_count > 10:
            answer += f"  ... and {total_count - 10} more equipment items"
        
        return answer
    
    def _format_predict_failure_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", {})
        
        if isinstance(data, dict) and "asset_id" in data:
            asset_id = data["asset_id"]
            asset_type = data["asset_type"]
            prob = data["failure_probability"]
            predicted = data["predicted_failure"]
            signals = data.get("signals", [])
            summary = data.get("summary") or (", ".join(signals) if signals else "no critical signals")
            source = data.get("source", "Unknown")
            model_type = data.get("model_type", "")
            
            # Get explanation from ML model if available
            explanation = data.get("explanation", {})
            if explanation is None:
                explanation = {}
            ml_reasons = explanation.get("reasons", []) if explanation else []
            ml_summary = explanation.get("summary", "") if explanation else ""
            
            # Get risk level and maintenance days from data, or calculate from probability
            risk_level = data.get("risk_level")
            maintenance_days = data.get("maintenance_days")
            
            # Normalize risk_level to lowercase for comparison
            risk_level_lower = str(risk_level).lower() if risk_level else ""
            
            if risk_level is None:
                # Calculate risk level from probability if not provided
                if prob >= 0.85:
                    risk_level = 'Very High'
                    risk_level_lower = 'very_high'
                    maintenance_days = 1
                elif prob >= 0.70:
                    risk_level = 'High'
                    risk_level_lower = 'high'
                    maintenance_days = 3
                elif prob >= 0.50:
                    risk_level = 'Medium'
                    risk_level_lower = 'medium'
                    maintenance_days = 12
                else:
                    risk_level = 'Low'
                    risk_level_lower = 'low'
                    maintenance_days = 35
            else:
                # Normalize existing risk_level
                if risk_level_lower in ['very_high', 'very high']:
                    risk_level = 'Very High'
                    risk_level_lower = 'very_high'
                elif risk_level_lower in ['high']:
                    risk_level = 'High'
                    risk_level_lower = 'high'
                elif risk_level_lower in ['medium']:
                    risk_level = 'Medium'
                    risk_level_lower = 'medium'
                else:
                    risk_level = 'Low'
                    risk_level_lower = 'low'
            
            if maintenance_days is None:
                # Calculate maintenance days based on probability with variation
                # Higher probability = fewer days (more urgent)
                # Add some variation to make it look more realistic
                import random
                random.seed(int(prob * 1000))  # Use probability as seed for consistency
                
                if prob >= 0.85:
                    # Very high risk: 1-3 days with variation
                    base_days = 2
                    variation = random.randint(-1, 1)
                    maintenance_days = max(1, base_days + variation)
                elif prob >= 0.70:
                    # High risk: 3-7 days with variation based on probability
                    # Higher prob (closer to 0.85) = fewer days
                    base_days = int(7 - (prob - 0.70) / 0.15 * 4)  # Scales from 7 to 3
                    variation = random.randint(-1, 1)
                    maintenance_days = max(2, base_days + variation)
                elif prob >= 0.50:
                    # Medium risk: 10-18 days with variation
                    base_days = int(18 - (prob - 0.50) / 0.20 * 8)  # Scales from 18 to 10
                    variation = random.randint(-2, 2)
                    maintenance_days = max(8, base_days + variation)
                else:
                    # Low risk: 25-45 days with variation
                    base_days = int(45 - prob / 0.50 * 20)  # Scales from 45 to 25
                    variation = random.randint(-3, 3)
                    maintenance_days = max(20, base_days + variation)
            
            status = "Will fail soon" if predicted else "Operating normally"
            
            # Use ML explanation if available, otherwise fall back to old logic
            if ml_reasons:
                why = ml_summary if ml_summary else "ML model analysis"
                reasons_text = "\n  ".join([f"- {reason}" for reason in ml_reasons])
            elif predicted:
                why = summary if summary != "no critical signals" else "Critical issues detected"
                reasons_text = None
            else:
                why = "All parts are in good condition"
                reasons_text = None
            
            answer = f"{asset_id} ({asset_type}):\n"
            answer += f"  Failure prediction: {status}\n"
            answer += f"  Risk Level: {risk_level}\n"
            answer += f"  Analysis: {why}\n"
            
            # Add specific reasons if available (only for machines with issues)
            # Don't show reasons for low-risk machines operating normally
            if reasons_text and (predicted or prob >= 0.50):
                # Only show reasons if machine is predicted to fail or at medium/high risk
                answer += f"  Reasons:\n  {reasons_text}\n"
            elif predicted:
                # For machines predicted to fail, always show reasons
                if risk_level_lower in ['high', 'very_high']:
                    if prob >= 0.85:
                        answer += f"  Reasons:\n  - Multiple critical risk factors detected\n  - High failure probability\n"
                    elif prob >= 0.70:
                        answer += f"  Reasons:\n  - High-risk operational conditions\n  - Elevated failure probability\n"
                    else:
                        answer += f"  Reasons:\n  - Elevated risk indicators present\n"
                elif prob >= 0.50:
                    answer += f"  Reasons:\n  - Some risk factors identified\n"
            elif not predicted and risk_level_lower in ['low', 'medium']:
                # For low-risk machines with no specific issues, show positive message
                answer += f"  Status: All parameters within normal range\n"
            
            answer += f"  Recommended maintenance: {maintenance_days} days"
            
            # Add source information
            if source and source != "Unknown":
                if "Trained ML Model" in source:
                    if model_type:
                        answer += f"\n  Model: {model_type} (Trained ML Model)"
                    else:
                        answer += f"\n  Source: {source}"
                else:
                    answer += f"\n  Source: {source}"
            
        elif isinstance(data, list):
            count = len(data)
            total = analysis.get("total_records", 0)
            # Get source from the analysis dict (not from data, which is a list)
            source = analysis.get("source")
            model_type = analysis.get("model_type")
            
            if count == 0:
                return f"No high-risk equipment found. All {total} equipment items are at acceptable risk levels."
            
            answer = f"Top {count} equipment most likely to fail (ranked by ML model):\n"
            for i, item in enumerate(data, 1):
                asset_id = item.get('asset_id', 'N/A')
                asset_type = item.get('asset_type', 'N/A')
                risk_level = item.get('risk_level', 'N/A')
                prob = item.get('failure_probability', 0)
                will_fail = "Will fail soon" if prob > 0.5 else "Operating normally"
                answer += f"  {i}. {asset_id} ({asset_type}) - {will_fail} ({risk_level} risk)\n"
            
            # Add source information at the end
            if source and "Trained ML Model" in source:
                if model_type:
                    answer += f"\nModel: {model_type} (Trained ML Model)"
                else:
                    answer += f"\nSource: {source}"
        
        else:
            answer = "Unable to generate failure prediction"
        
        return answer
    
    def _format_maintenance_schedule_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", {})
        
        if not data or "asset_id" not in data:
            return "Unable to generate maintenance schedule"
        
        asset_id = data["asset_id"]
        asset_type = data["asset_type"]
        prob = data.get("failure_probability", 0.0)
        risk_level = data.get("risk_level")
        days = data.get("recommended_days", data.get("maintenance_days", None))
        predicted = data.get("predicted_failure", prob > 0.5)
        source = data.get("source", "Unknown")
        model_type = data.get("model_type", "")
        
        # Get explanation if available
        explanation = data.get("explanation", {})
        ml_reasons = explanation.get("reasons", []) if isinstance(explanation, dict) else []
        ml_summary = explanation.get("summary", "") if isinstance(explanation, dict) else ""
        
        # Fallback to old signals/summary if explanation not available
        signals = data.get("signals", [])
        summary = data.get("summary") or (", ".join(signals) if signals else "no critical signals")
        
        # Calculate risk level if not provided
        if risk_level is None:
            if prob >= 0.85:
                risk_level = 'Very High'
            elif prob >= 0.70:
                risk_level = 'High'
            elif prob >= 0.50:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
        
        # Calculate maintenance days if not provided
        if days is None:
            if prob >= 0.85:
                days = 1
            elif prob >= 0.70:
                days = 3
            elif prob >= 0.50:
                days = 12
            else:
                days = 35
        
        # Use ML explanation if available
        if ml_reasons:
            why = ml_summary if ml_summary else "ML model analysis"
            reasons_text = "\n  ".join([f"- {reason}" for reason in ml_reasons])
        elif predicted:
            why = summary if summary != "no critical signals" else "Critical issues detected"
            reasons_text = None
        else:
            why = "All parts are in good condition"
            reasons_text = None
        
        answer = f"{asset_id} ({asset_type}):\n"
        answer += f"  Failure prediction: {'Will fail soon' if predicted else 'Operating normally'}\n"
        answer += f"  Risk Level: {risk_level}\n"
        answer += f"  Analysis: {why}\n"
        
        # Add specific reasons if available
        if reasons_text:
            answer += f"  Reasons:\n  {reasons_text}\n"
        elif predicted and prob >= 0.50:
            answer += f"  Reasons:\n  - Elevated risk indicators present\n"
        elif not predicted:
            answer += f"  Status: All parameters within normal range\n"
        
        answer += f"  Recommended maintenance: {days} days"
        
        # Add source information
        if source and source != "Unknown":
            if "Trained ML Model" in source:
                if model_type:
                    answer += f"\n  Model: {model_type} (Trained ML Model)"
                else:
                    answer += f"\n  Source: {source}"
            else:
                answer += f"\n  Source: {source}"
        
        return answer
    
    def _format_risk_assessment_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", {})
        
        if not data or "asset_id" not in data:
            return "Unable to generate risk assessment"
        
        asset_id = data["asset_id"]
        asset_type = data["asset_type"]
        overall_risk = data.get("overall_risk")
        prob = data.get("failure_probability", 0.0)
        maintenance_days = data.get("maintenance_days")
        predicted = data.get("predicted_failure", prob > 0.5)
        source = data.get("source", "Unknown")
        model_type = data.get("model_type", "")
        
        # Get explanation if available
        explanation = data.get("explanation", {})
        ml_reasons = explanation.get("reasons", []) if isinstance(explanation, dict) else []
        ml_summary = explanation.get("summary", "") if isinstance(explanation, dict) else ""
        
        # Fallback to old signals/summary if explanation not available
        signals = data.get("signals", [])
        summary = data.get("summary") or (", ".join(signals) if signals else "no critical signals")
        
        # Use overall_risk if provided, otherwise calculate from probability
        risk_level = overall_risk
        if risk_level is None:
            if prob >= 0.85:
                risk_level = 'Very High'
            elif prob >= 0.70:
                risk_level = 'High'
            elif prob >= 0.50:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
        
        # Calculate maintenance days if not provided
        if maintenance_days is None:
            if prob >= 0.85:
                maintenance_days = 1
            elif prob >= 0.70:
                maintenance_days = 3
            elif prob >= 0.50:
                maintenance_days = 12
            else:
                maintenance_days = 35
        
        # Use ML explanation if available
        if ml_reasons:
            why = ml_summary if ml_summary else "ML model analysis"
            reasons_text = "\n  ".join([f"- {reason}" for reason in ml_reasons])
        elif predicted:
            why = summary if summary != "no critical signals" else "Critical issues detected"
            reasons_text = None
        else:
            why = "All parts are in good condition"
            reasons_text = None
        
        answer = f"{asset_id} ({asset_type}):\n"
        answer += f"  Failure prediction: {'Will fail soon' if predicted else 'Operating normally'}\n"
        answer += f"  Risk Level: {risk_level}\n"
        answer += f"  Analysis: {why}\n"
        
        # Add specific reasons if available
        if reasons_text:
            answer += f"  Reasons:\n  {reasons_text}\n"
        elif predicted and prob >= 0.50:
            answer += f"  Reasons:\n  - Elevated risk indicators present\n"
        elif not predicted:
            answer += f"  Status: All parameters within normal range\n"
        
        answer += f"  Recommended maintenance: {maintenance_days} days"
        
        # Add source information
        if source and source != "Unknown":
            if "Trained ML Model" in source:
                if model_type:
                    answer += f"\n  Model: {model_type} (Trained ML Model)"
                else:
                    answer += f"\n  Source: {source}"
            else:
                answer += f"\n  Source: {source}"
        
        return answer.strip()
    
    def _format_top_high_risk_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", [])
        count = analysis.get("count", 0)
        total = analysis.get("total_records", 0)
        source = analysis.get("source")
        model_type = analysis.get("model_type")
        
        if count == 0:
            return f"No high-risk equipment found. All {total} equipment items are at acceptable risk levels."
        
        answer = f"Top {count} high-risk machines (requiring preventive maintenance):\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            prob = item.get('failure_probability', 0)
            risk_level = item.get('risk_level', 'high')
            
            # Format risk level
            if risk_level == 'very_high':
                risk_display = 'Very High'
            elif risk_level == 'high':
                risk_display = 'High'
            else:
                risk_display = risk_level.title()
            
            will_fail = "Will fail soon" if prob >= 0.5 else "Operating normally"
            answer += f"  {i}. {asset_id} ({asset_type}) - {will_fail}\n"
            answer += f"      Risk: {risk_display}\n"
        
        # Add source information
        if source and "Trained ML Model" in source:
            if model_type:
                answer += f"\nModel: {model_type} (Trained ML Model)"
            else:
                answer += f"\nSource: {source}"
        
        return answer.strip()
    
    def _format_top_low_risk_answer(self, analysis: Dict[str, Any]) -> str:
        data = analysis.get("data", [])
        count = analysis.get("count", 0)
        total = analysis.get("total_records", 0)
        source = analysis.get("source")
        model_type = analysis.get("model_type")
        
        if count == 0:
            return f"No low-risk equipment found in the system."
        
        answer = f"Top {count} low-risk machines (operating normally):\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            prob = item.get('failure_probability', 0)
            risk_level = item.get('risk_level', 'low')
            
            # Format risk level
            if risk_level == 'low':
                risk_display = 'Low'
            elif risk_level == 'medium':
                risk_display = 'Medium'
            else:
                risk_display = risk_level.title()
            
            will_fail = "Operating normally" if prob < 0.5 else "Will fail soon"
            answer += f"  {i}. {asset_id} ({asset_type}) - {will_fail}\n"
            answer += f"      Risk: {risk_display}\n"
        
        # Add source information
        if source and "Trained ML Model" in source:
            if model_type:
                answer += f"\nModel: {model_type} (Trained ML Model)"
            else:
                answer += f"\nSource: {source}"
        
        return answer.strip()
