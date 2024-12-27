from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Initialize the AnswerRelevancyMetric with a threshold of 0.5
arm = AnswerRelevancyMetric(threshold=0.5)

# Define multiple test cases with different examples

test_case1 = LLMTestCase(
    input="What is the price of the latest iPhone Pro model?",
    actual_output="The iPhone 15 Pro model starts at $999.",
    retrieval_context=["The iPhone 15 Pro model is priced from $999."]
)

test_case2 = LLMTestCase(
    input="How much does an iPhone SE cost?",
    actual_output="The iPhone SE is available from $429.",
    retrieval_context=["The starting price for the iPhone SE is $429."]
)

test_case3 = LLMTestCase(
    input="Can you tell me the cost of an iPhone 14?",
    actual_output="The iPhone 14 starts at $799.",
    retrieval_context=["The iPhone 14 has a starting price of $799."]
)

test_case4 = LLMTestCase(
    input="What is the cost of an iPhone 13 Mini?",
    actual_output="The iPhone 13 Mini starts at $699.",
    retrieval_context=["The iPhone 13 Mini starts at a price of $699."]
)

test_case5 = LLMTestCase(
    input="How much would I pay for the iPhone 12?",
    actual_output="The iPhone 12 is priced starting at $799.",
    retrieval_context=["The iPhone 12 has a starting price of $799."]
)

# Evaluate all test cases using the AnswerRelevancyMetric
evaluate([test_case1, test_case2, test_case3, test_case4, test_case5], metrics=[arm])
