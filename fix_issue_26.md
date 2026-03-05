# Implementation of `parameter_extraction_f1()` Evaluation Metric

## Summary

This pull request implements the `parameter_extraction_f1()` function in `smithery/eval/param_extraction.py` to measure the precision, recall, and F1 score for a model's ability to extract the correct arguments for tool calls. The metric assesses the correctness of key-value pairs in the tool call arguments, thus aiding in evaluating how accurately a model extracts parameters for the tool usage.

## Implementation

```python
from typing import List, Dict, Any, NamedTuple

class ToolCall(NamedTuple):
    name: str
    arguments: Dict[str, Any]

class ParamMetrics(NamedTuple):
    precision: float
    recall: float
    f1: float
    type_accuracy: float

def find_matching_call(pred_calls: List[ToolCall], tool_name: str) -> ToolCall:
    for call in pred_calls:
        if call.name == tool_name:
            return call
    return None

def parameter_extraction_f1(
    predicted: List[List[ToolCall]],
    expected: List[List[ToolCall]]
) -> ParamMetrics:
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_type_correct = 0
    total_params = 0
    count = 0
    
    for pred_calls, exp_calls in zip(predicted, expected):
        for exp_call in exp_calls:
            matching_pred = find_matching_call(pred_calls, exp_call.name)
            if matching_pred is None:
                continue
            
            exp_args = set(exp_call.arguments.items())
            pred_args = set(matching_pred.arguments.items())
            
            true_positives = len(exp_args & pred_args)
            precision = true_positives / len(pred_args) if pred_args else 0
            recall = true_positives / len(exp_args) if exp_args else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1
            
            for key in exp_call.arguments:
                if key in matching_pred.arguments:
                    total_params += 1
                    if type(exp_call.arguments[key]) == type(matching_pred.arguments[key]):
                        total_type_correct += 1
    
    precision_avg = total_precision / count if count else 0
    recall_avg = total_recall / count if count else 0
    f1_avg = total_f1 / count if count else 0
    type_accuracy = total_type_correct / total_params if total_params else 0
    
    return ParamMetrics(precision=precision_avg, recall=recall_avg, f1=f1_avg, type_accuracy=type_accuracy)
```

## Test Cases

```python
def test_parameter_extraction_f1():
    predicted = [
        [ToolCall(name="get_weather", arguments={"city": "Tokyo", "units": "fahrenheit"})]
    ]
    
    expected = [
        [ToolCall(name="get_weather", arguments={"city": "Tokyo", "units": "celsius"})]
    ]
    
    metrics = parameter_extraction_f1(predicted, expected)
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.type_accuracy == 1.0  # Assuming correct types

    print("Test cases passed successfully.")

test_parameter_extraction_f1()
```

## Explanation of Changes

1. Implemented the `parameter_extraction_f1()` function that iterates through the predicted and expected tool call lists.
2. For each tool call, precision, recall, and F1 scores are calculated based on the correctness of key-value pairs.
3. Incorporated an accuracy check for the types of arguments with corresponding type accuracy measurement.
4. Added test cases to validate the correctness of the implementation.

This function now effectively evaluates how accurately parameters are extracted in terms of both data and type correctness.