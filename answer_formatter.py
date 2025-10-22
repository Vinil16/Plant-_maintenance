"""
Answer Formatter - Fixed and Optimized
Provides correct, formatted responses for all question types
"""

from typing import Dict, Any

class AnswerFormatter:
    def __init__(self):
        """Initialize answer formatter with units mapping."""
        self.units = {
            'temperature': '°C',
            'vibration_level': 'mm/s',
            'pressure': 'bar',
            'runtime_hours': 'hours'
        }
    
    def format_answer(self, analysis: Dict[str, Any], parsed_question: Dict[str, Any]) -> str:
        """Format analysis result into human-readable answer."""
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
        else:
            return f"{analysis.get('summary', 'Analysis completed')}"
    
    def _format_count_answer(self, analysis: Dict[str, Any]) -> str:
        """Format count answer."""
        count = analysis.get("data", 0)
        equipment = analysis.get("equipment")
        total_records = analysis.get("total_records", 0)
        
        if equipment:
            return f"There are {count} {equipment.lower()}s in the system (out of {total_records} total equipment items)."
        else:
            return f"There are {count} total equipment items in the system."
    
    def _format_average_answer(self, analysis: Dict[str, Any]) -> str:
        """Format average answer."""
        value = analysis.get("data", 0)
        column = analysis.get("column", "")
        unit = self.units.get(column, "")
        total_records = analysis.get("total_records", 0)
        
        column_name = column.replace('_', ' ').title()
        return f"The average {column_name} is {value} {unit} (calculated from {total_records} equipment items)."
    
    def _format_max_answer(self, analysis: Dict[str, Any]) -> str:
        """Format maximum answer."""
        value = analysis.get("data", 0)
        column = analysis.get("column", "")
        unit = self.units.get(column, "")
        total_records = analysis.get("total_records", 0)
        
        column_name = column.replace('_', ' ').title()
        return f"The highest {column_name} is {value} {unit} (from {total_records} equipment items)."
    
    def _format_min_answer(self, analysis: Dict[str, Any]) -> str:
        """Format minimum answer."""
        value = analysis.get("data", 0)
        column = analysis.get("column", "")
        unit = self.units.get(column, "")
        total_records = analysis.get("total_records", 0)
        
        column_name = column.replace('_', ' ').title()
        return f"The lowest {column_name} is {value} {unit} (from {total_records} equipment items)."
    
    def _format_top_answer(self, analysis: Dict[str, Any]) -> str:
        """Format top N answer."""
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
        """Format show outdated answer."""
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
        """Format show high risk answer."""
        data = analysis.get("data", [])
        count = analysis.get("count", 0)
        total_records = analysis.get("total_records", 0)
        
        if count == 0:
            return f"No high-risk equipment found. All {total_records} equipment items are at acceptable risk levels."
        
        answer = f"Found {count} high-risk equipment items (out of {total_records} total):\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            risk = item.get('failure_probability', 0)
            answer += f"  {i}. {asset_id} - {asset_type} (Risk: {risk:.2f})\n"
        
        if count > 10:
            answer += f"  ... and {count - 10} more high-risk items"
        
        return answer.strip()
    
    def _format_show_equipment_answer(self, analysis: Dict[str, Any]) -> str:
        """Format show equipment answer."""
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
        """Format show all answer."""
        data = analysis.get("data", [])
        total_count = analysis.get("total_records", 0)
        
        answer = f"Showing {len(data)} of {total_count} total equipment items:\n"
        for i, item in enumerate(data, 1):
            asset_id = item.get('asset_id', 'N/A')
            asset_type = item.get('asset_type', 'N/A')
            answer += f"  {i}. {asset_id} - {asset_type}\n"
        
        if total_count > 10:
            answer += f"  ... and {total_count - 10} more equipment items"
        
        return answer.strip()