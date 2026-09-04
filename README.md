<!--
Copyright 2026 Intrinsic Innovation LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# intrinsic-inference

## Overview

This repository contains code for ml inference applications.

> [!NOTE]
> This repository is currently a preview release for iterations with the Special
Interest Group on Physical AI and is not considered stable.

## Structure

```text
intrinsic-inference/
├── core/                     # Core components for managing models and running inference through a backend (i.e. Triton).
│   ├── BUILD
│   ├── v1/                   # Shared, non-framework specific proto definitions.
│   │   ├── BUILD
│   │   └── ml_model.proto
│   ├── inference_runner.py
│   └── ...
└── ros/                      # ROS related code.
    ├── BUILD
    ├── inference_msgs        # Inference node interface package.
    │   └── ...
    └── inference_node        # ROS inference node.
        ├── BUILD
        ├── inference_node.py
        └── ...
```

## Docs

-   [**ROS inference node docs**](intrinsic_inference/ros/inference_node/README.md)
-   [**Core framework docs**](intrinsic_inference/core/README.md)
