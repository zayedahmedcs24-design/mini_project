# System Architecture & Technical Specifications

**Course:** B.M.S. College of Engineering, UG CSE 5th-Semester Mini-Project  
**Title:** Uncertainty-Aware Flood Extent Mapping from Sentinel-1 SAR with Cross-Region Evaluation and Exposure Estimation  
**Anchor Milestone:** Review-2 (Design with Basic Module Implementation)

---

## 1. End-to-End System Architecture

```mermaid
graph TD
    subgraph DataIngestion ["1. Data Ingestion & Radiometric Preprocessing"]
        A[Sentinel-1 SAR VV/VH GeoTIFFs] --> B[Decibel Verification & Range Check]
        B --> C[NaN Imputation & Physical Clipping]
        C --> D[Channel-Wise Z-Score Normalization]
        L[Hand-Labeled Ground Truth Masks] --> M[Ignore Masking: Filter -1 Pixels]
    end

    subgraph ModelingEngine ["2. Segmentation & Inference Engine"]
        D --> N[PyTorch DataLoader]
        M --> N
        N --> E[Classical Baseline: Otsu Thresholding]
        N --> F[Deep ConvNet: U-Net ResNet-34]
        N --> G[Deep ConvNet: DeepLabV3+ ASPP]
        N --> H[Transformer: SegFormer-B0]
    end

    subgraph UncertaintyEngine ["3. Uncertainty Quantification & Calibration"]
        F --> I[Monte Carlo Dropout T=20 Passes]
        G --> J[Deep Ensembles 3 Independent Seeds]
        I --> K[Predictive Mean Probability]
        I --> O[Predictive Entropy & Mutual Information]
        J --> K
        O --> P[Calibration: ECE & Reliability Diagrams]
        O --> Q[Validation: Error-vs-Uncertainty AUROC]
    end

    subgraph ExposureEngine ["4. Geospatial Exposure Engine"]
        K --> R[Dynamic UTM Zone Estimation per Chip]
        R --> S[Fractional WorldPop Aggregation: Exposed Population]
        R --> T[OpenStreetMap Vector Intersection: Flooded Road km]
        O --> U[Uncertainty Brackets: Conservative / Expected / Worst-Case]
    end

    subgraph Deployment ["5. Web Dashboard & Serving"]
        K --> V[FastAPI Backend Service]
        U --> V
        V --> W[Interactive Leaflet Geo-Explorer]
    end

    style DataIngestion fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style ModelingEngine fill:#ede7f6,stroke:#512da8,stroke-width:2px
    style UncertaintyEngine fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style ExposureEngine fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style Deployment fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

---

## 2. Complete Data Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Disaster Analyst / Guide
    participant Client as Web Frontend (Leaflet.js)
    participant API as FastAPI Backend
    participant Storage as Precomputed Results Cache
    participant Engine as PyTorch Inference Engine
    participant GIS as Geospatial Exposure Engine

    User->>Client: Selects Flood Event (e.g. Bolivia Held-Out Event)
    Client->>API: GET /api/chips?split=bolivia
    API->>Storage: Read chip bounding boxes & metadata
    Storage-->>API: Chip metadata JSON
    API-->>Client: Returns chip list & bounds

    User->>Client: Toggles Uncertainty & Exposure Layers
    Client->>API: GET /api/chips/{chip_id}/layers
    API->>Storage: Fetch precomputed RGBA PNG tiles
    Storage-->>API: Stream tiles (SAR, Mask, Entropy, Error)
    API-->>Client: Render synchronized dual-panel split map

    Client->>API: GET /api/chips/{chip_id}/exposure
    API->>GIS: Query population & road length intervals
    GIS-->>API: Return [Conservative, Expected, Worst-Case]
    API-->>Client: Render Exposure KPI Cards

    opt Single Chip Live Upload (Could-Have)
        User->>Client: Uploads local GeoTIFF chip
        Client->>API: POST /predict (multipart/form-data)
        API->>Engine: Run normalized forward pass (T=10 MC passes)
        Engine->>GIS: Compute fractional exposure on CPU
        GIS-->>API: Predicted mask + uncertainty + exposure
        API-->>Client: Display live diagnostic overlay (<5s)
    end
```

---

## 3. Module Boundaries & Engineering Responsibilities

```mermaid
classDiagram
    class DataPipeline {
        +fetch_and_verify_splits()
        +download_chips()
        +normalize_sar()
        +get_class_balance()
    }
    class EvaluationMetrics {
        +compute_confusion_matrix()
        +compute_metrics_from_counts()
        +evaluate_predictions()
        +MetricTracker
    }
    class SegmentationModels {
        +FloodUNet
        +FloodDeepLabV3Plus
        +otsu_segment_chip()
        +GlobalOtsuBaseline
    }
    class UncertaintyEngine {
        +mc_dropout_inference()
        +compute_predictive_entropy()
        +compute_mutual_information()
        +compute_ece()
    }
    class ExposureEngine {
        +estimate_utm_crs()
        +compute_fractional_population()
        +compute_flooded_road_length()
    }

    DataPipeline --> SegmentationModels : Feeds normalized tensors
    SegmentationModels --> EvaluationMetrics : Evaluates ignore-aware metrics
    SegmentationModels --> UncertaintyEngine : Samples stochastic passes
    UncertaintyEngine --> ExposureEngine : Provides risk thresholds
```
