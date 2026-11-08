"""
Lightweight FLUX service schema tests — no torch/CUDA required.
Validates Python syntax and Pydantic schema correctness only.
These run on the CI test job (ubuntu-latest, no GPU) before the Docker build.
"""
import ast
import pathlib
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError


# ── Mirror the schemas from app.py for isolated validation ───────────────────
# (app.py imports torch at module level; we validate the schema independently)

class ReferenceInput(BaseModel):
    role: str
    asset_id: str
    character_id: Optional[str] = None
    label: str
    image_base64: str


class GenerateRequest(BaseModel):
    prompt: str
    width: int  = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=576,  ge=64, le=2048)
    output_format: str = 'png'
    seed: int   = Field(default=42)
    references: List[ReferenceInput] = Field(default_factory=list)


# ── Syntax ────────────────────────────────────────────────────────────────────

def test_app_python_syntax():
    src = pathlib.Path('services/flux/app.py').read_text(encoding='utf-8')
    tree = ast.parse(src)
    assert tree is not None, 'app.py failed Python syntax check'


def test_requirements_contain_expected_packages():
    reqs = pathlib.Path('services/flux/requirements.txt').read_text(encoding='utf-8')
    for pkg in ('torch', 'diffusers', 'fastapi', 'uvicorn', 'pydantic', 'Pillow'):
        assert pkg in reqs, f'requirements.txt missing: {pkg}'


# ── Schema ────────────────────────────────────────────────────────────────────

def test_generate_request_defaults():
    req = GenerateRequest(prompt='test prompt')
    assert req.width == 1024
    assert req.height == 576
    assert req.seed == 42
    assert req.references == []
    assert req.output_format == 'png'


def test_generate_request_vertical_pilot011():
    req = GenerateRequest(prompt='test', width=576, height=1024, seed=99)
    assert req.width == 576
    assert req.height == 1024
    assert req.seed == 99


def test_generate_request_rejects_invalid_dimensions():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt='test', width=10)   # below ge=64
    with pytest.raises(ValidationError):
        GenerateRequest(prompt='test', height=9999) # above le=2048


def test_generate_request_with_references():
    ref = ReferenceInput(
        role='character', asset_id='pilot-011-cole-hayes',
        label='Cole Hayes', image_base64='aGVsbG8=',
    )
    req = GenerateRequest(prompt='test', references=[ref])
    assert len(req.references) == 1
    assert req.references[0].role == 'character'
    assert req.references[0].character_id is None


def test_reference_roles():
    for role in ('character', 'location', 'prop'):
        ref = ReferenceInput(
            role=role, asset_id=f'pilot-011-{role}',
            label=role, image_base64='dGVzdA==',
        )
        assert ref.role == role


def test_reference_with_character_id():
    ref = ReferenceInput(
        role='character', asset_id='pilot-011-cole-hayes',
        character_id='cole-hayes', label='Cole Hayes', image_base64='dGVzdA==',
    )
    assert ref.character_id == 'cole-hayes'
