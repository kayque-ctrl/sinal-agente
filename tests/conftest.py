"""Fixtures compartilhadas entre os testes."""

from pathlib import Path

import pytest

PASTA_DADOS = Path(__file__).parent / "dados"


@pytest.fixture
def pasta_dados() -> Path:
    """Pasta com os casos e prompts de exemplo do agente de vendas."""
    return PASTA_DADOS
