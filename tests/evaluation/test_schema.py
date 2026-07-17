def test_evaluation_package_is_not_public_library_api() -> None:
    import cite_right
    import evaluation

    assert evaluation.DATASET_VERSION == "1.0.0"
    assert not hasattr(cite_right, "evaluation")
