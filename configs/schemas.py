from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetParamsModel(BaseModel):
    preprocessed: bool = False


class ExperimentSettingsModel(BaseModel):
    cv_folds: int = Field(ge=2)
    test_size: float = Field(gt=0, lt=1)
    random_state: int = 42
    priority_metrics: List[str] = Field(min_length=1)
    selected_classifiers: List[str] = Field(min_length=1)

    @field_validator('priority_metrics', 'selected_classifiers')
    @classmethod
    def no_empty_values(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError('list must contain at least one non-empty value')
        return cleaned


class BenchmarkExperimentModel(BaseModel):
    methods: List[str] = Field(min_length=1)
    datasets: List[str] = Field(min_length=1)
    datasets_params: DatasetParamsModel = DatasetParamsModel()
    experiment_config: ExperimentSettingsModel

    @field_validator('methods', 'datasets')
    @classmethod
    def normalize_names(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError('list must contain at least one non-empty value')
        return cleaned


class MethodDefinitionModel(BaseModel):
    source: Literal['smote_variants', 'local'] = 'smote_variants'
    class_name: str = Field(alias='class', min_length=1)
    module: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('module')
    @classmethod
    def module_required_for_local(cls, value: Optional[str], info):
        source = info.data.get('source')
        if source == 'local' and (value is None or not value.strip()):
            raise ValueError("'module' is required when source is 'local'")
        return value


class DatasetDefinitionModel(BaseModel):
    data_id: Optional[str] = None
    prep_data_id: Optional[str] = None


class MethodsRegistryModel(BaseModel):
    methods: Dict[str, MethodDefinitionModel]


class DatasetsRegistryModel(BaseModel):
    datasets: Dict[str, DatasetDefinitionModel]
