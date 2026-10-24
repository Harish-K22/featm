"""Examples execute in isolated interpreters so registrations cannot leak."""
from pathlib import Path
import re
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def text(path):
    p=ROOT/path
    if not p.exists():pytest.skip('Repository documentation not packaged here')
    return p.read_text(encoding='utf-8')


def execute(code):
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)


def test_authoring_examples_and_public_stateful_import():
    s=text('docs/FEATURE_AUTHORING.md')
    blocks=re.findall(r'```python\n(.*?)```',s,re.S)
    assert len(blocks)==6
    execute(blocks[0]+'\nnp.testing.assert_allclose(mean_abs([[-1.,2.]]),[1.5])')
    execute(blocks[1]+'\nnp.testing.assert_allclose(descriptor([[[1.,3.]]]).values,[[2.,1.,3.]])')
    execute(blocks[4]+'\nassert issubclass(MyEncoder, StatefulEncoder)')
    assert 'from featm.state import StatefulEncoder' in blocks[4]
    assert 'from featm.state import _stateful_adapter' in blocks[5]
    # Placeholders and registration dependent on an implemented class are labeled.
    assert s.count('schematic')>=4
    assert 'preserve partition order' in blocks[4]


@pytest.mark.parametrize('path',['README.md','docs/API.md','docs/FEATURE_METADATA.md'])
def test_public_python_examples(path):
    for block in re.findall(r'```python\n(.*?)```',text(path),re.S):execute(block)


def test_compatibility_history_and_pypi_link():
    s=text('docs/COMPATIBILITY.md')
    for version in ('0.5.7','0.5.8','0.5.9','0.5.10','0.5.11'):assert version in s
    assert 'without expanding that certification scope' in s
    assert 'https://github.com/Harish-K22/featm/blob/main/docs/FEATURE_METADATA.md' in text('README.md')
