"""Exercise an installed wheel in a fresh environment, never an editable install."""
import os,subprocess,sys,tempfile,venv
from pathlib import Path

root=Path(__file__).resolve().parents[2]
wheel=next(Path(sys.argv[1]).resolve().glob('*.whl'))
env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
with tempfile.TemporaryDirectory() as tmp:
 base=Path(tmp);v=base/'env';venv.create(v,with_pip=True)
 py=v/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
 def run(*args):subprocess.run([str(py),*map(str,args)],cwd=base,env=env,check=True)
 # These are test dependencies, not changes to the package's runtime metadata.
 run('-m','pip','install',wheel,'pytest','networkx','scipy','pyarrow')
 run('-m','pip','check')
 run('-c',"import featm,importlib.metadata as m,numpy as np,sys; from pathlib import Path; assert Path(featm.__file__).resolve().is_relative_to(Path(sys.prefix)); s=featm.REGISTRY.all(); assert featm.__version__==m.version('featm')=='0.5.7'; assert [len(s),len({x.modality for x in s}),len({(x.modality,x.family) for x in s}),sum(x.stateful for x in s),sum(x.gpu_available for x in s)]==[552,18,182,7,22]; np.testing.assert_allclose(featm.extract(np.array([[1.,2.,3.,4.],[10.,20.,30.,40.]]),features=['timeseries.statistical.mean'],errors='raise').to_numpy().ravel(),[2.5,25.]); print('INSTALLED_WHEEL',featm.__file__)")
 cfg=base/'pytest.ini';cfg.write_text('[pytest]\n')
 runner=base/'runner.py';runner.write_text("import os,sys\nfrom pathlib import Path\nroot=Path(sys.argv.pop(1)); sys.path.insert(0,str(root)); os.chdir(root)\nimport featm,pytest\nassert not Path(featm.__file__).resolve().is_relative_to(root)\nraise SystemExit(pytest.main(sys.argv[1:]))\n")
 run(runner,root,'-c',cfg,'tests/test_core.py','tests/test_state.py','tests/test_matrix.py','tests/test_v057_checkpoint_root.py','-q','-p','no:cacheprovider')
