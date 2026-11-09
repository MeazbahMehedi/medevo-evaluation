# primary on test questions

Status: complete

| Dataset | Type | Generated | Accuracy % | 95% CI | Boxed % | Truncated | s/question |
|---|---|---:|---:|---|---:|---:|---:|
| vqa_rad | all | 200/200 | 15.5 | 11.0-20.5 | 98.0 | 5 | 23.9 |
| vqa_rad | open | 89/89 | 15.7 | 9.0-23.6 | 95.5 | 5 | 26.8 |
| vqa_rad | closed | 111/111 | 15.3 | 9.0-22.5 | 100.0 | 0 | 21.6 |
| slake | all | 200/200 | 27.5 | 21.5-33.5 | 99.5 | 1 | 22.1 |
| slake | open | 122/122 | 27.9 | 20.5-36.1 | 99.2 | 1 | 22.8 |
| slake | closed | 78/78 | 26.9 | 17.9-37.2 | 100.0 | 0 | 21.1 |

Overall: 21.5% (86/400), boxed 98.8%.

Primary metric: boxed-format-unwrapped-em-v1:normalized_correct. Other scorer/field combinations are in summary.json.
