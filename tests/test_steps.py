import pytest

from service.models import Classification, StructuredFields
from service.steps.abstraction import compute_abstraction_flag
from service.steps.classifier import DummyClassifier


@pytest.mark.asyncio
async def test_dummy_classifier_multiple_categories() -> None:
    classifier = DummyClassifier()
    result = await classifier.classify("I have urgent billing question")
    assert result == [Classification.URGENT, Classification.ADMIN]


@pytest.mark.asyncio
async def test_dummy_classifier_emergent() -> None:
    classifier = DummyClassifier()
    result = await classifier.classify("This is an emergency situation")
    assert result == [Classification.EMERGENT]


@pytest.mark.asyncio
async def test_dummy_classifier_urgent() -> None:
    classifier = DummyClassifier()
    result = await classifier.classify("I need urgent help")
    assert result == [Classification.URGENT]


@pytest.mark.asyncio
async def test_dummy_classifier_admin() -> None:
    classifier = DummyClassifier()
    result = await classifier.classify("I have a billing question")
    assert result == [Classification.ADMIN]


@pytest.mark.asyncio
async def test_dummy_classifier_routine_default() -> None:
    classifier = DummyClassifier()
    result = await classifier.classify("I have a mild headache")
    assert result == [Classification.ROUTINE]


def test_abstraction_flag_emergent() -> None:
    fields = StructuredFields()
    assert compute_abstraction_flag([Classification.EMERGENT], fields) is True


def test_abstraction_flag_urgent_severe() -> None:
    fields = StructuredFields(severity="severe")
    assert compute_abstraction_flag([Classification.URGENT], fields) is True


def test_abstraction_flag_urgent_not_severe() -> None:
    fields = StructuredFields(severity="mild")
    assert compute_abstraction_flag([Classification.URGENT], fields) is False


def test_abstraction_flag_routine() -> None:
    fields = StructuredFields()
    assert compute_abstraction_flag([Classification.ROUTINE], fields) is False
