import pandas as pd
import numpy as np
from typing import Dict, Any

class DataAnalyzer:
    """Analyzes plant data and executes queries based on parsed questions."""
    
    def __init__(self, csv_file: str):
        # Try different encodings to load CSV file
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                self.df = pd.read_csv(csv_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if not hasattr(self, 'df'):
            raise ValueError("Could not load CSV file")
        
        self.columns = list(self.df.columns)

        # Calculate equipment age from install date
        if 'install_date' in self.columns:
            try:
                install_dt = pd.to_datetime(self.df['install_date'], errors='coerce')
                age_days = (pd.Timestamp.now() - install_dt).dt.days
                self.df['equipment_age_days'] = age_days.fillna(age_days.median())
            except Exception:
                pass

        # Calculate runtime intensity (hours per day)
        if 'runtime_hours' in self.columns and 'equipment_age_days' in self.df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.df['runtime_intensity'] = self.df['runtime_hours'] / (self.df['equipment_age_days'] + 1)

        # Build index for fast asset lookup
        self._asset_index = {}
        if 'asset_id' in self.columns:
            for i, aid in enumerate(self.df['asset_id']):
                if isinstance(aid, str):
                    self._asset_index[aid.upper()] = i

        # Cache high-risk equipment for faster queries
        self._high_risk_cache = None
        if 'failure_probability' in self.columns:
            try:
                high_risk_df = self.df[self.df['failure_probability'] > 0.7]
                result_columns = ['asset_id', 'asset_type', 'failure_probability']
                available_columns = [c for c in result_columns if c in self.columns]
                self._high_risk_cache = high_risk_df[available_columns].sort_values(
                    by=available_columns[-1] if 'failure_probability' in available_columns else available_columns[0],
                    ascending=False
                )
            except Exception:
                self._high_risk_cache = None

        # Enable fuzzy matching for asset IDs if rapidfuzz is available
        try:
            from rapidfuzz import process
            self._fuzzy_available = True
            self._all_asset_ids = list(self._asset_index.keys())
        except Exception:
            self._fuzzy_available = False
            self._all_asset_ids = list(self._asset_index.keys())
    
    def analyze_data(self, parsed_question: Dict[str, Any]) -> Dict[str, Any]:
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
            elif analysis_type == "predict_failure":
                return self._analyze_predict_failure(parameters)
            elif analysis_type == "maintenance_schedule":
                return self._analyze_maintenance_schedule(parameters)
            elif analysis_type == "risk_assessment":
                return self._analyze_risk_assessment(parameters)
            elif analysis_type == "top_high_risk":
                return self._handle_top_high_risk(parameters.get("count", 10))
            elif analysis_type == "top_low_risk":
                return self._handle_top_low_risk(parameters.get("count", 10))
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {str(e)}"}
    
    def _handle_count(self, equipment_type: str) -> Dict[str, Any]:
        if equipment_type and 'asset_type' in self.columns:
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
            return {
                "success": True, 
                "data": len(self.df), 
                "type": "count",
                "total_records": len(self.df)
            }
    
    def _handle_average(self, column: str) -> Dict[str, Any]:
        if not column:
            return {"success": False, "error": "No column specified for average calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
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
        if not column:
            return {"success": False, "error": "No column specified for max calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
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
        if not column:
            return {"success": False, "error": "No column specified for min calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
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
        if not column:
            return {"success": False, "error": "No column specified for top N calculation"}
        
        if column not in self.columns:
            return {"success": False, "error": f"Column '{column}' not found in dataset"}
        
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            return {"success": False, "error": f"Column '{column}' is not numeric"}
        
        top_data = self.df.nlargest(count, column)
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
        if 'model_status' not in self.columns:
            return {"success": False, "error": "No 'model_status' column found in dataset"}
        
        outdated_df = self.df[self.df['model_status'].str.contains('Obsolete', case=False, na=False)]
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
        if 'failure_probability' not in self.columns:
            return {"success": False, "error": "No 'failure_probability' column found in dataset"}
        
        if self._high_risk_cache is not None:
            high_risk_df = self._high_risk_cache
        else:
            high_risk_df = self.df[self.df['failure_probability'] > 0.7]
        
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
        if not equipment_type:
            return {"success": False, "error": "No equipment type specified"}
        
        if 'asset_type' not in self.columns:
            return {"success": False, "error": "No 'asset_type' column found in dataset"}
        
        equipment_df = self.df[self.df['asset_type'].str.contains(equipment_type, case=False, na=False)]
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
    
    def _analyze_predict_failure(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ml_predictor import MaintenancePredictor
            
            predictor = MaintenancePredictor()
            requested_id = parameters.get("asset_id")
            asset_id = self._resolve_asset_id(requested_id) if requested_id else None
            
            if asset_id and asset_id in self._asset_index:
                result = predictor.predict_failure(asset_id=asset_id, df=self.df)
                if "error" in result:
                    return {
                        "success": False,
                        "error": result["error"]
                    }
                else:
                    # Get asset details from dataframe
                    asset_row = self.df[self.df.get("asset_id", pd.Series()) == asset_id]
                    asset_type = asset_row.iloc[0].get("asset_type", "Unknown") if not asset_row.empty else "Unknown"
                    
                    return {
                        "success": True,
                        "data": {
                            "asset_id": asset_id,
                            "asset_type": asset_type,
                            "predicted_failure": result.get("predicted_failure", False),
                            "failure_probability": result.get("ml_predicted_probability", result.get("failure_probability", 0.0)),
                            "risk_level": result.get("risk_level", "low"),
                            "explanation": result.get("explanation", {}),
                            "source": "Trained ML Model",
                        },
                        "type": "predict_failure",
                        "total_records": len(self.df)
                    }
            elif not requested_id:
                # Get high-risk equipment using batch prediction
                limit = parameters.get("limit", 10)
                try:
                    # Convert dataframe rows to dicts for batch prediction
                    assets_list = self.df.to_dict('records')
                    predictions_df = predictor.batch_predict(assets_list)
                    
                    # Filter high-risk (probability >= 0.7) and sort
                    high_risk = predictions_df[predictions_df['failure_probability'] >= 0.7].sort_values(
                        'failure_probability', ascending=False
                    ).head(limit)
                    
                    high_risk_list = []
                    for idx, row in high_risk.iterrows():
                        high_risk_list.append({
                            "asset_id": row.get("asset_id", "Unknown"),
                            "asset_type": row.get("asset_type", "Unknown"),
                            "failure_probability": float(row.get("failure_probability", 0.0)),
                            "predicted_failure": bool(row.get("predicted_failure", False)),
                            "risk_level": row.get("risk_level", "high")
                        })
                    
                    return {
                        "success": True,
                        "data": high_risk_list,
                        "type": "predict_failure",
                        "count": len(high_risk_list),
                        "total_records": len(self.df),
                        "source": "ML Model",
                        "model_type": "Extra Trees"
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to get high-risk equipment: {str(e)}"
                    }
            else:
                suggestions = self._suggest_asset_ids(requested_id)
                hint = f" Similar IDs: {', '.join(suggestions)}" if suggestions else ""
                return {"success": False, "error": f"Asset {requested_id} not found.{hint}"}
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction analysis failed: {str(e)}"
            }
    
    def _analyze_maintenance_schedule(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ml_predictor import MaintenancePredictor
            
            predictor = MaintenancePredictor()
            requested_id = parameters.get("asset_id")
            asset_id = self._resolve_asset_id(requested_id) if requested_id else None
            
            if asset_id and asset_id in self._asset_index:
                result = predictor.get_maintenance_schedule(asset_id, self.df)
                if result.get("success"):
                    return {
                        "success": True,
                        "data": result,
                        "type": "maintenance_schedule",
                        "asset_id": asset_id,
                        "total_records": len(self.df)
                    }
                else:
                    return result
            else:
                if requested_id:
                    suggestions = self._suggest_asset_ids(requested_id)
                    hint = f" Similar IDs: {', '.join(suggestions)}" if suggestions else ""
                    return {
                        "success": False,
                        "error": f"Asset {requested_id} not found.{hint}"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Asset ID required for maintenance schedule"
                    }
        except Exception as e:
            return {
                "success": False,
                "error": f"Maintenance schedule analysis failed: {str(e)}"
            }
    
    def _analyze_risk_assessment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from ml_predictor import MaintenancePredictor
            
            predictor = MaintenancePredictor()
            asset_id = parameters.get("asset_id")
            asset_id = self._resolve_asset_id(asset_id) if asset_id else None
            
            if asset_id:
                result = predictor.assess_risk_level(asset_id, self.df)
                if result.get("success"):
                    return {
                        "success": True,
                        "data": result,
                        "type": "risk_assessment",
                        "asset_id": asset_id,
                        "total_records": len(self.df)
                    }
                else:
                    return result
            else:
                return {
                    "success": False,
                    "error": "Asset ID required for risk assessment"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Risk assessment analysis failed: {str(e)}"
            }

    def _handle_top_high_risk(self, count: int) -> Dict[str, Any]:
        """Get top N high-risk machines using ML predictions."""
        try:
            from ml_predictor import MaintenancePredictor
            
            predictor = MaintenancePredictor()
            
            # Convert dataframe rows to dicts for batch prediction
            assets_list = self.df.to_dict('records')
            predictions_df = predictor.batch_predict(assets_list)
            
            # Filter high-risk machines (risk_level: very_high or high, or probability >= 0.70)
            high_risk = predictions_df[
                (predictions_df['risk_level'].isin(['very_high', 'high'])) | 
                (predictions_df['failure_probability'] >= 0.70)
            ].sort_values('failure_probability', ascending=False).head(count)
            
            high_risk_list = []
            for idx, row in high_risk.iterrows():
                high_risk_list.append({
                    "asset_id": row.get("asset_id", "Unknown"),
                    "asset_type": row.get("asset_type", "Unknown"),
                    "failure_probability": float(row.get("failure_probability", 0.0)),
                    "predicted_failure": bool(row.get("predicted_failure", False)),
                    "risk_level": row.get("risk_level", "high")
                })
            
            return {
                "success": True,
                "data": high_risk_list,
                "type": "top_high_risk",
                "count": len(high_risk_list),
                "total_records": len(self.df),
                "source": "Trained ML Model",
                "model_type": "Extra Trees"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get top high-risk machines: {str(e)}"
            }
    
    def _handle_top_low_risk(self, count: int) -> Dict[str, Any]:
        """Get top N low-risk machines using ML predictions."""
        try:
            from ml_predictor import MaintenancePredictor
            
            predictor = MaintenancePredictor()
            
            # Convert dataframe rows to dicts for batch prediction
            assets_list = self.df.to_dict('records')
            predictions_df = predictor.batch_predict(assets_list)
            
            # Filter low-risk machines (risk_level: low, or probability < 0.50)
            low_risk = predictions_df[
                (predictions_df['risk_level'] == 'low') | 
                (predictions_df['failure_probability'] < 0.50)
            ].sort_values('failure_probability', ascending=True).head(count)
            
            low_risk_list = []
            for idx, row in low_risk.iterrows():
                low_risk_list.append({
                    "asset_id": row.get("asset_id", "Unknown"),
                    "asset_type": row.get("asset_type", "Unknown"),
                    "failure_probability": float(row.get("failure_probability", 0.0)),
                    "predicted_failure": bool(row.get("predicted_failure", False)),
                    "risk_level": row.get("risk_level", "low")
                })
            
            return {
                "success": True,
                "data": low_risk_list,
                "type": "top_low_risk",
                "count": len(low_risk_list),
                "total_records": len(self.df),
                "source": "Trained ML Model",
                "model_type": "Extra Trees"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get top low-risk machines: {str(e)}"
            }

    def _resolve_asset_id(self, asset_id: str) -> str:
        if not asset_id:
            return None
        query = asset_id.upper().strip()

        candidates = [query]
        if " " in query:
            candidates.append(query.replace(" ", "-"))
            candidates.append(query.replace(" ", ""))

        import re
        m = re.match(r"^([A-Z]+)[- ]?(\d+)$", query)
        if m:
            prefix, num = m.group(1), m.group(2)
            zero_padded = f"{prefix}-{int(num):03d}"
            candidates.extend([
                f"{prefix}{int(num):03d}",
                zero_padded,
                f"{prefix}-{int(num)}",
                f"{prefix}{int(num)}",
            ])

        seen = set()
        uniq_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq_candidates.append(c)

        for cand in uniq_candidates:
            if cand in self._asset_index:
                return cand

        if self._fuzzy_available and self._all_asset_ids:
            try:
                from rapidfuzz import process, fuzz
                match, score, _ = process.extractOne(query, self._all_asset_ids, scorer=fuzz.WRatio)
                if match and score >= 80:
                    return match
            except Exception:
                pass

        return asset_id

    def _suggest_asset_ids(self, query_id: str, limit: int = 3):
        try:
            from rapidfuzz import process, fuzz
            if not self._all_asset_ids:
                return []
            query = str(query_id).upper().strip()
            results = process.extract(query, self._all_asset_ids, scorer=fuzz.WRatio, limit=limit)
            return [r[0] for r in results if r and len(r) >= 2]
        except Exception:
            return []
