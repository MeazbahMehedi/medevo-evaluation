# adapted on test questions

Status: complete

| Dataset | Type | Generated | Accuracy % | 95% CI | Boxed % | Truncated | s/question |
|---|---|---:|---:|---|---:|---:|---:|
| vqa_rad | all | 200/200 | 51.0 | 44.0-58.0 | 99.5 | 1 | 21.8 |
| vqa_rad | open | 89/89 | 24.7 | 15.7-33.7 | 98.9 | 1 | 23.8 |
| vqa_rad | closed | 111/111 | 72.1 | 64.0-80.2 | 100.0 | 0 | 20.2 |
| slake | all | 200/200 | 49.0 | 42.0-56.0 | 99.5 | 4 | 22.6 |
| slake | open | 122/122 | 29.5 | 21.3-37.7 | 99.2 | 3 | 24.0 |
| slake | closed | 78/78 | 79.5 | 70.5-88.5 | 100.0 | 1 | 20.3 |

Overall: 50.0% (200/400), boxed 99.5%.

Primary metric: boxed-format-unwrapped-em-v1:normalized_correct. Other scorer/field combinations are in summary.json.
