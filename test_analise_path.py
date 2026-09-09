from pathlib import Path
import importlib.util


def test_analise_root_points_to_workspace_root():
    spec = importlib.util.spec_from_file_location(
        "analise_module",
        Path("Otimizador/analise/analise.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected_root = Path("c:/Repositorios/TCC")
    assert module.ROOT == expected_root
    assert module.DATA_DIR == expected_root / "Output_dados" / "ml_preparado"
    assert module.MODEL_DIR == expected_root / "Output_dados" / "resultados_xgboost"
