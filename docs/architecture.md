# Architecture

## Training Pipeline

```mermaid
flowchart LR
  A["C-MAPSS FD001-FD004"] --> B["Regime-aware preprocessing"]
  B --> C["Patch Transformer candidates"]
  C --> D["Physics-guided ablations"]
  D --> E["Regime-consistent backbone"]
  E --> F["Uncertainty and policy refinement"]
  E --> G["KAN experimental branch"]
  F --> H["Critical-boundary safety guard"]
```

## Frozen Production Inference Pipeline

```mermaid
flowchart LR
  A["Engine history CSV"] --> B["Input validation"]
  B --> C["Regime-aware preprocessing"]
  C --> D["Physics-guided Patch Transformer"]
  D --> E["Critical-boundary safety guard"]
  E --> F["Conformal intervals"]
  F --> G["Maintenance policy"]
  G --> H["Structured prediction response"]
```

## Physics-Guided Transformer

```mermaid
flowchart TB
  A["Sensor sequence"] --> B["10x5 temporal patches"]
  B --> C["Transformer encoder"]
  C --> D["Mean pooled latent state"]
  D --> E["RUL head"]
  C --> F["Regime-consistency training signal"]
```

## Safety Guard

```mermaid
flowchart LR
  A["Base RUL"] --> B{"15 < base <= 25?"}
  B -- "yes" --> C["Apply downward correction"]
  B -- "no" --> D["Leave unchanged"]
  C --> E["Safety-adjusted RUL"]
  D --> E
```

## Uncertainty And Maintenance Flow

```mermaid
flowchart LR
  A["Safety-adjusted RUL"] --> B["Global split conformal radii"]
  B --> C["80/90/95 intervals"]
  C --> D["Support and review logic"]
  D --> E["Urgent, schedule, inspect, or monitor"]
```

## Experimental KAN Branch

```mermaid
flowchart LR
  A["Frozen backbone residuals"] --> B["Engineering features"]
  B --> C["KAN and non-KAN residual candidates"]
  C --> D["Global AeroKAN experiment"]
  C --> E["Selective one-sided AeroKAN experiment"]
  D --> F["Not selected"]
  E --> F
```

## Monitoring Architecture

```mermaid
flowchart LR
  A["Inference requests"] --> B["Structured logs"]
  B --> C["Schema and range checks"]
  B --> D["Prediction and interval drift"]
  B --> E["Guard and review rates"]
  B --> F["Latency and failures"]
```

Selected production path: preprocessing, Regime-Consistent Physics-Guided Patch Transformer, deterministic guard, conformal intervals, and maintenance policy.

Experimental paths: global and selective AeroKAN residual correction branches.
