import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
#data set was cleaned
class WeatherPrediction(BaseModel):
    class_name: str = Field(..., alias="className")  
    confidence: float
    is_correct: Optional[bool] = Field(None, alias="isCorrect")

class WeatherPredictionResult(BaseModel):
    predictions: List[WeatherPrediction]
    predicted_label: str = Field(..., alias="predictedLabel")
    actual_label: Optional[str] = Field(None, alias="actualLabel")
    accuracy: Optional[float] = None
    processing_time: float = Field(..., alias="processingTime")
    confidence_threshold: Optional[float] = Field(None, alias="confidenceThreshold")

class ModelInfo(BaseModel):
    model_name: str = Field(default="VGG16 Transfer Learning", alias="modelName")
    backbone: str = Field(default="VGG16")
    pretrained: bool = True
    num_classes: int = Field(default=4)
    dataset_size: int = Field(default=1125)
    test_accuracy: float = Field(default=0.9823, alias="testAccuracy")

class WeatherAnalysisMetadata(BaseModel):
    analysis_id: str = Field(..., alias="analysisId")
    timestamp: datetime.datetime
    image_name: str = Field(..., alias="imageName")
    image_size: Optional[str] = Field(None, alias="imageSize")
    dataset_split: Optional[str] = Field(None, alias="datasetSplit")  

class WeatherAnalysisResult(WeatherAnalysisMetadata):
    model_info: ModelInfo = Field(..., alias="modelInfo")
    results: WeatherPredictionResult

class WeatherBatchAnalysis(BaseModel):
    batch_id: str = Field(..., alias="batchId")
    timestamp: datetime.datetime
    total_images: int = Field(..., alias="totalImages")
    overall_accuracy: Optional[float] = Field(None, alias="overallAccuracy")
    analyses: List[WeatherAnalysisResult]
