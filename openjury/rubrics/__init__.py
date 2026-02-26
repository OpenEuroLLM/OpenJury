"""Rubrics package for structured LLM evaluation.

Modules:
    - ``openjury.rubrics.schema``: dataclasses for rubric definitions and scores
    - ``openjury.rubrics.defaults``: built-in rubric registry (currently ``default``)
    - ``openjury.rubrics.io``: custom rubric loading/registration from JSON
    - ``openjury.rubrics.scorer``: rubric scoring with an LLM judge
    - ``openjury.rubrics.pipeline``: shared rubric pipeline output helpers

Prefer explicit imports from submodules in this package.
"""
