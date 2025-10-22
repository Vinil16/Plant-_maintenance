"""
Data Analyzer - Fixed and Optimized
Handles all analysis types correctly with proper error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class DataAnalyzer:
    def __init__(self, csv_file: str):
        """Initialize with dataset and proper error handling."""
        # Load CSV with multiple encoding support
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                self.df = pd.read_csv(csv_file, encoding=encoding)
                print(f"Loaded {len(self.df)} records with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if not hasattr(self, 'df'):
            raise ValueError("Could not load CSV file")
        
        self.columns = list(self.df.columns)
        print(f"Available columns: {self.columns}")
    
    def analyze_data(self, parsed_question: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data based on parsed question with comprehensive error handling."""
        analysis_type = parsed_question.get("analysis_type", "")
        parameters = parsed_question.get("parameters", {})
        
        try:
            if analysis_type == "count":
                return self._handle_count(parameters.get("equipment_type"))
            elif analysis_type == "average":
                return self._handle_average(parameters.get("column"))
            elif analysis_type == "max":
                return self._handle_max(parameters.get("column"))
            elif analysis_type == "min":
                return self._handle_min(parameters.get("column"))
            elif analysis_type == "top":
                return self._handle_top(parameters.get("column"), parameters.get("count", 5))
            elif analysis_type == "show_outdated":
                return self._handle_show_outdated()
            elif analysis_type == "show_high_risk":
                return self._handle_show_high_risk()
            elif analysis_type == "show_equipment":
                return self._handle_show_equipment(parameters.get("equipment_type"))
            elif analysis_type == "show_all":
                return self._handle_show_all()
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {str(e)}"}
    
    def _handle_count(self, equipment_type: str) -> Dict[str, Any]:
        """Handle count operations with proper equipment filtering."""
        if equipment_type and 'asset_type' in self.columns:
            # Filter by equipment type (case-insensitive)
            filtered_df = self.df[self.df['asset_type'].str.contains(equipment_type, case=False, na=False)]
            count = len(filtered_df)
            return {
                "success": True, 
                "data": count, 
                "type": "count", 
                "equipment": equipment_type,
                "total_records": len(self.df)
            }
        else:
            # Count all equipment
            return {
                "success": True, 
                "data": len(self.df), 
                "type": "count",
                "total_records": len(self.df)
            }
    
    def _handle_average(self, column: str) -> Dict[str, Any]:
        """Handle average operations with proper validation."""
        if not column:
            return {"success": False, "error": "No column specified for average calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
        # Calculate average, handling NaN values
        avg_value = self.df[column].mean()
        if pd.isna(avg_value):
            return {"success": False, "error": f"No valid numeric data in column '{column}'"}
        
        return {
            "success": True, 
            "data": round(avg_value, 2), 
            "type": "average", 
            "column": column,
            "total_records": len(self.df)
        }
    
    def _handle_max(self, column: str) -> Dict[str, Any]:
        """Handle max operations with proper validation."""
        if not column:
            return {"success": False, "error": "No column specified for max calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
        # Calculate max, handling NaN values
        max_value = self.df[column].max()
        if pd.isna(max_value):
            return {"success": False, "error": f"No valid numeric data in column '{column}'"}
        
        return {
            "success": True, 
            "data": round(max_value, 2), 
            "type": "max", 
            "column": column,
            "total_records": len(self.df)
        }
    
    def _handle_min(self, column: str) -> Dict[str, Any]:
        """Handle min operations with proper validation."""
        if not column:
            return {"success": False, "error": "No column specified for min calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
        # Calculate min, handling NaN values
        min_value = self.df[column].min()
        if pd.isna(min_value):
            return {"success": False, "error": f"No valid numeric data in column '{column}'"}
        
        return {
            "success": True, 
            "data": round(min_value, 2), 
            "type": "min", 
            "column": column,
            "total_records": len(self.df)
        }
    
    def _handle_top(self, column: str, count: int) -> Dict[str, Any]:
        """Handle top N operations with proper validation."""
        if not column:
            return {"success": False, "error": "No column specified for top N calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
        # Get top N rows sorted by column in descending order
        top_data = self.df.nlargest(count, column)
        
        # Prepare result with available columns
        result_columns = ['asset_id', 'asset_type', column]
        available_columns = [col for col in result_columns if col in self.df.columns]
        
        result = top_data[available_columns].to_dict('records')
        
        return {
            "success": True, 
            "data": result, 
            "type": "top", 
            "column": column, 
            "count": count,
            "total_records": len(self.df)
        }
    
    def _handle_show_outdated(self) -> Dict[str, Any]:
        """Handle show outdated operations."""
        if 'model_status' not in self.columns:
            return {"success": False, "error": "No 'model_status' column found in dataset"}
        
        # Filter for obsolete/outdated equipment
        outdated_df = self.df[self.df['model_status'].str.contains('Obsolete', case=False, na=False)]
        
        # Prepare result with available columns
        result_columns = ['asset_id', 'asset_type', 'model_status']
        available_columns = [col for col in result_columns if col in self.df.columns]
        
        result = outdated_df[available_columns].head(10).to_dict('records')
        
        return {
            "success": True, 
            "data": result, 
            "type": "show", 
            "count": len(outdated_df),
            "total_records": len(self.df)
        }
    
    def _handle_show_high_risk(self) -> Dict[str, Any]:
        """Handle show high risk operations."""
        if 'failure_probability' not in self.columns:
            return {"success": False, "error": "No 'failure_probability' column found in dataset"}
        
        # Filter for high-risk equipment (probability > 0.7)
        high_risk_df = self.df[self.df['failure_probability'] > 0.7]
        
        # Prepare result with available columns
        result_columns = ['asset_id', 'asset_type', 'failure_probability']
        available_columns = [col for col in result_columns if col in self.df.columns]
        
        result = high_risk_df[available_columns].head(10).to_dict('records')
        
        return {
            "success": True, 
            "data": result, 
            "type": "show", 
            "count": len(high_risk_df),
            "total_records": len(self.df)
        }
    
    def _handle_show_equipment(self, equipment_type: str) -> Dict[str, Any]:
        """Handle show equipment operations."""
        if not equipment_type:
            return {"success": False, "error": "No equipment type specified"}
        
        if 'asset_type' not in self.columns:
            return {"success": False, "error": "No 'asset_type' column found in dataset"}
        
        # Filter by equipment type (case-insensitive)
        equipment_df = self.df[self.df['asset_type'].str.contains(equipment_type, case=False, na=False)]
        
        # Prepare result with available columns
        result_columns = ['asset_id', 'asset_type']
        available_columns = [col for col in result_columns if col in self.df.columns]
        
        result = equipment_df[available_columns].head(10).to_dict('records')
        
        return {
            "success": True, 
            "data": result, 
            "type": "show", 
            "count": len(equipment_df),
            "equipment": equipment_type,
            "total_records": len(self.df)
        }
    
    def _handle_show_all(self) -> Dict[str, Any]:
        """Handle show all operations."""
        # Prepare result with available columns
        result_columns = ['asset_id', 'asset_type']
        available_columns = [col for col in result_columns if col in self.df.columns]
        
        result = self.df[available_columns].head(10).to_dict('records')
        
        return {
            "success": True, 
            "data": result, 
            "type": "show", 
            "count": len(self.df),
            "total_records": len(self.df)
        }