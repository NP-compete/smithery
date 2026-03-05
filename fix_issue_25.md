# Implement `tool_selection_accuracy()` Evaluation Metric

## Summary

This pull request implements the `tool_selection_accuracy()` function, along with an extended version that provides detailed per-tool accuracy and confusion data. Additional test cases have also been included to ensure comprehensive coverage of the function's behavior.

## Changes

1. **Core Function**: Implemented the `tool_selection_accuracy()` function to compute the overall accuracy of tool selection.
2. **Per-tool Accuracy**: Computes accuracy per tool, reflecting how well the model performs for each specific tool.
3. **Confusion Data**: Gathers data on which tools are most often confused with each other.
4. **Test Cases**: Added comprehensive test cases to validate the implementation.

### Implementation

```python
from typing import List, Dict, Tuple
from collections import defaultdict

class ToolCall:
    def __init__(self, name: str):
        self.name = name

class ToolAccuracyReport:
    def __init__(self, overall_accuracy: float, per_tool_accuracy: Dict[str, float], confusion_pairs: List[Tuple[str, str, int]]):
        self.overall_accuracy = overall_accuracy
        self.per_tool_accuracy = per_tool_accuracy
        self.confusion_pairs = confusion_pairs

def tool_selection_accuracy(
    predicted: List[List[ToolCall]],
    expected: List[List[ToolCall]],
) -> ToolAccuracyReport:
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must have the same length")
    
    if not predicted:
        return ToolAccuracyReport(0.0, {}, [])
    
    correct = 0
    tool_correct_counts = defaultdict(int)
    tool_total_counts = defaultdict(int)
    confusion_counts = defaultdict(lambda: defaultdict(int))

    for pred_calls, exp_calls in zip(predicted, expected):
        pred_names = {c.name for c in pred_calls}
        exp_names = {c.name for c in exp_calls}
        
        # Track per-tool counts
        for tool in exp_names:
            tool_total_counts[tool] += 1
            if tool in pred_names:
                tool_correct_counts[tool] += 1
        
        if pred_names == exp_names:
            correct += 1
        else:
            for exp in exp_names:
                for pred in pred_names:
                    if exp != pred:
                        confusion_counts[exp][pred] += 1

    overall_accuracy = correct / len(predicted)
    per_tool_accuracy = {tool: tool_correct_counts[tool] / tool_total_counts[tool] for tool in tool_total_counts}
    confusion_pairs = [(exp, pred, count) for exp, preds in confusion_counts.items() for pred, count in preds.items()]

    return ToolAccuracyReport(overall_accuracy, per_tool_accuracy, confusion_pairs)
```

### Test Cases

```python
import unittest

class TestToolSelectionAccuracy(unittest.TestCase):
    
    def test_perfect_accuracy(self):
        predicted = [[ToolCall('A')], [ToolCall('B')]]
        expected = [[ToolCall('A')], [ToolCall('B')]]
        result = tool_selection_accuracy(predicted, expected)
        self.assertEqual(result.overall_accuracy, 1.0)
        
    def test_zero_accuracy(self):
        predicted = [[ToolCall('A')], [ToolCall('B')]]
        expected = [[ToolCall('B')], [ToolCall('A')]]
        result = tool_selection_accuracy(predicted, expected)
        self.assertEqual(result.overall_accuracy, 0.0)
    
    def test_partial_accuracy(self):
        predicted = [[ToolCall('A')], [ToolCall('A')]]
        expected = [[ToolCall('A')], [ToolCall('B')]]
        result = tool_selection_accuracy(predicted, expected)
        self.assertEqual(result.overall_accuracy, 0.5)
    
    def test_multi_tool_order_insensitive(self):
        predicted = [[ToolCall('A'), ToolCall('B')]]
        expected = [[ToolCall('B'), ToolCall('A')]]
        result = tool_selection_accuracy(predicted, expected)
        self.assertEqual(result.overall_accuracy, 1.0)
    
    def test_per_tool_accuracy(self):
        predicted = [[ToolCall('A')], [ToolCall('B')], [ToolCall('C')]]
        expected = [[ToolCall('A')], [ToolCall('B')], [ToolCall('A')]]
        result = tool_selection_accuracy(predicted, expected)
        self.assertAlmostEqual(result.per_tool_accuracy['A'], 0.5)
        self.assertAlmostEqual(result.per_tool_accuracy['B'], 1.0)

if __name__ == '__main__':
    unittest.main()
```

## Explanation

This implementation focuses on measuring the accuracy of tool selection in a flexible manner. It handles various scenarios such as multi-tool orders and provides insights into individual tool performance and tool confusion. The accompanied test cases aim to ensure robustness and correctness of the logic across different situations.