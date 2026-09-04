# Copyright 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tensor metadata describing input or output tensor schemas

string name
string datatype     # OIP string (e.g., "FP32", "INT64") for OIP familiarity
uint8 dtype_code    # DLPack code (e.g., 2 for float)
uint16 dtype_bits   # DLPack bit width (e.g., 32)
int64[] shape
