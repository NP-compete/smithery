# Pull Request: Implement `validate_dataset()` for Schema Validation

## Summary

This PR implements the `validate_dataset()` function in `smithery/data/validators.py`. The function is designed to validate training examples against predefined tool schemas and generate a comprehensive validation report. This is crucial for ensuring data integrity before model training, as any invalid data can adversely affect the model's performance.

## Implementation Details

The `validate_dataset()` function checks each training example for:
1. The existence of tool names.
2. The presence and correctness of required parameters.
3. Parameter type matching.
4. Validity of enumerated values.
5. Absence of extra unknown parameters.
6. The detection of duplicate examples.

### Code Implementation

```python
from collections import Counter
from smithery.data.models import ValidationReport, TrainingExample, ToolDefinition

def validate_dataset(
    examples: list[TrainingExample],
    tools: list[ToolDefinition],
) -> ValidationReport:
    tool_names = {t.name for t in tools}
    tool_map = {t.name: t for t in tools}
    errors = []
    tool_counts = Counter()
    valid = 0
    seen_examples = set()
    duplicate_count = 0
    
    for i, ex in enumerate(examples):
        example_valid = True
        tool_call_data = (tuple(ex.messages), tuple(ex.tool_calls))
        
        # Check 6: detect duplicates
        if tool_call_data in seen_examples:
            duplicate_count += 1
            errors.append(f"Example {i}: duplicate example detected")
            example_valid = False
        else:
            seen_examples.add(tool_call_data)
        
        for call in ex.tool_calls:
            # Check 1: tool name exists
            if call.name not in tool_names:
                errors.append(f"Example {i}: unknown tool '{call.name}'")
                example_valid = False
                continue
            
            tool = tool_map[call.name]
            tool_counts[call.name] += 1
            
            # Validate parameters
            tool_params = tool.parameters
            call_params = call.parameters
            
            # Check 2: required params present
            missing_params = [p for p in tool_params if tool_params[p]["required"] and p not in call_params]
            if missing_params:
                errors.append(
                    f"Example {i}: tool '{call.name}' missing required parameters {missing_params}"
                )
                example_valid = False
            
            # Check 3: param types match
            for p, value in call_params.items():
                if p not in tool_params:
                    errors.append(f"Example {i}: tool '{call.name}' has unknown parameter '{p}'")
                    example_valid = False
                    continue
                
                expected_type = tool_params[p]["type"]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Example {i}: tool '{call.name}', parameter '{p}' expected type {expected_type.__name__}, found {type(value).__name__}"
                    )
                    example_valid = False
                
            # Check 4: enum values valid
            for p, value in call_params.items():
                if "enum" in tool_params[p]:
                    allowed_values = tool_params[p]["enum"]
                    if value not in allowed_values:
                        errors.append(
                            f"Example {i}: tool '{call.name}', parameter '{p}' has invalid value '{value}'"
                        )
                        example_valid = False
        
        if example_valid:
            valid += 1

    return ValidationReport(
        total_count=len(examples),
        valid_count=valid,
        invalid_count=len(examples) - valid,
        duplicate_count=duplicate_count,
        tool_balance=dict(tool_counts),
        errors=errors,
    )
```

### Test Cases

```python
import unittest
from smithery.data.models import TrainingExample, ToolDefinition, ToolCall

class TestValidateDataset(unittest.TestCase):
    
    def setUp(self):
        # Define tool definitions with parameters
        self.tool_definitions = [
            ToolDefinition(
                name="tool_a",
                parameters={
                    "param1": {"type": str, "required": True},
                    "param2": {"type": int, "required": False, "enum": [1, 2, 3]}
                }
            ),
            ToolDefinition(
                name="tool_b",
                parameters={
                    "param1": {"type": bool, "required": True}
                }
            )
        ]
        
    def test_validate_valid_example(self):
        examples = [
            TrainingExample(
                messages=['Hello'],
                tool_calls=[ToolCall(name='tool_a', parameters={"param1": "value", "param2": 2})]
            )
        ]
        report = validate_dataset(examples, self.tool_definitions)
        self.assertEqual(report.valid_count, 1)
        self.assertEqual(report.invalid_count, 0)
        self.assertEqual(report.duplicate_count, 0)
        self.assertEqual(report.errors, [])
    
    def test_validate_missing_parameter(self):
        examples = [
            TrainingExample(
                messages=['Hello'],
                tool_calls=[ToolCall(name='tool_a', parameters={"param2": 2})]
            )
        ]
        report = validate_dataset(examples, self.tool_definitions)
        self.assertEqual(report.valid_count, 0)
        self.assertEqual(report.invalid_count, 1)
        self.assertIn("missing required parameters", report.errors[0])
    
    def test_validate_unknown_tool(self):
        examples = [
            TrainingExample(
                messages=['Hello'],
                tool_calls=[ToolCall(name='unknown_tool', parameters={})]
            )
        ]
        report = validate_dataset(examples, self.tool_definitions)
        self.assertEqual(report.valid_count, 0)
        self.assertEqual(report.invalid_count, 1)
        self.assertIn("unknown tool", report.errors[0])
        
    def test_validate_type_mismatch(self):
        examples = [
            TrainingExample(
                messages=['Hello'],
                tool_calls=[ToolCall(name='tool_a', parameters={"param1": 123})]
            )
        ]
        report = validate_dataset(examples, self.tool_definitions)
        self.assertEqual(report.valid_count, 0)
        self.assertEqual(report.invalid_count, 1)
        self.assertIn("expected type str, found int", report.errors[0])
    
    def test_duplicate_example(self):
        examples = [
            TrainingExample(
                messages=['Hello'],
                tool_calls=[ToolCall(name='tool_a', parameters={"param1": "value"})]
            ),
            TrainingExample(
                messages=['Hello'],
                tool_calls=[ToolCall(name='tool_a', parameters={"param1": "value"})]
            )
        ]
        report = validate_dataset(examples, self.tool_definitions)
        self.assertEqual(report.valid_count, 1)
        self.assertEqual(report.invalid_count, 1)
        self.assertEqual(report.duplicate_count, 1)
        
if __name__ == '__main__':
    unittest.main()
```

### Explanation of Changes

- **Function Implementation**: The `validate_dataset()` function is now implemented with comprehensive validation checks as per the schema.
- **Error Handling**: The function records specific, user-readable error messages for different types of validation failures.
- **Test Cases**: Test cases were added to verify the correctness of the validation logic against various potential input data scenarios.

This complete implementation provides a robust framework for validating training datasets based on specified tool schemas, thereby ensuring high-quality data for model training.