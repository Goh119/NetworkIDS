# NetworkIDS

An Explainable and Robust L2-L4 Header and Flow-Based Intrusion Detection System.

## Project Objectives

1. Extract L2-L4 header and flow features from network traffic.
2. Train a machine learning model for traffic classification.
3. Provide explainable predictions using SHAP.
4. Validate packet header integrity using rule-based checks.
5. Evaluate model robustness under controlled L2-L4 header perturbations.

## Technologies

- Python
- Scapy
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Matplotlib

## Src folder python files
1. config.py
2. packet_parser.py
3. flow_builder.py
4. label_matcher.py
5. dataset_validator.py
6. analyze_unmatched.py
7. prepare_training_data.py
8. analyze_duplicates.py
9. inspect_conflicting_duplicate.py
10. split_dataset.py
11. train_model.py --> model produced at here
12. shap_explain.py
13. analyze_prediction_errors.py
14. analyze_integrity_rules.py
15. analyze_l4_checksum.py
16. integrity_checker.py 
17. inspect_s2.py 可以删掉
18. analyze_integrity_results.py 
19. robustness_evaluation.py
20. robust_train_model.py
21. robust_train_model_v2.py