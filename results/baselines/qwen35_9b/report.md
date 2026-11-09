# baseline on test questions

Status: complete

| Dataset | Type | Generated | Accuracy % | 95% CI | Boxed % | Truncated | s/question |
|---|---|---:|---:|---|---:|---:|---:|
| vqa_rad | all | 200/200 | 56.5 | 49.5-63.5 | 94.0 | 14 | 42.0 |
| vqa_rad | open | 89/89 | 28.1 | 19.1-37.1 | 87.6 | 13 | 47.1 |
| vqa_rad | closed | 111/111 | 79.3 | 71.2-86.5 | 99.1 | 1 | 38.0 |
| slake | all | 200/200 | 57.5 | 50.5-64.0 | 95.5 | 14 | 39.4 |
| slake | open | 122/122 | 41.0 | 32.8-50.0 | 93.4 | 12 | 42.9 |
| slake | closed | 78/78 | 83.3 | 74.4-91.0 | 98.7 | 2 | 34.0 |

Overall: 57.0% (228/400), boxed 94.8%.

Primary metric: boxed-format-unwrapped-em-v1:normalized_correct. Other scorer/field combinations are in summary.json.
