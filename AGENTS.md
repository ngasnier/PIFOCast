# PIFOCast — AGENTS.md

## Setup
- `pixi install` pour créer l'environnement (pas pip/poetry).
- Dépendances système : `sudo apt install netcdf-bin proj-bin`. Si erreur `libproj.so.15` : `sudo ln -s libproj.so.25 libproj.so.15`.
- Téléchargement ERA5 via `cdsapi` : nécessite `~/.cdsapirc` avec une clé API CDS.

## Layout
- **Le vrai code** est dans `/app/pifocast/` (package importable).
- `/app/src/pifocast/` est un **template vide** laissé par pixi — ne pas ajouter de code là-dedans.
- L'ancien code TF/TF-GNN est archivé dans `/app/src-archive/pifocast/`.

## Workflow
- Piloté par notebooks : d'abord `pifocast-dataset.ipynb` (génération des données), puis `pifocast-graph.ipynb` (entraînement et test).
- `dataset/` → fichiers GRIB ERA5 ; `pifo_chk/` → checkpoints (tous deux gitignorés).
- Téléchargement en CLI : `python era5.py`.

## Tests / CI / Linting
- **Aucun test, CI, linter, formateur, ou vérificateur de types configuré.**
- `tests/` est vide (seulement `__init__.py`).

## Stack technique
- **JAX 0.5+** + **Flax 0.10+** + **Optax 0.2+** + **Orbax 0.2+**.
- Build : Hatchling. Python 3.11–3.13 (vérrouillé 3.12).
- MLP par défaut avec activation `silu`.

## Architecture modèle (résumé)
- `pifocast/model.py` : `PifoModel` (Flax `nn.Module`) — Encoder → Processor (4× message passing) → Decoder sur un graphe de grille unique.
- Entrée : 6 features par nœud (Z, U, V + cos_lat, cos_lon, sin_lon) ; 3 features par arête (longueur, coord_x, coord_y).
- Sortie : résidu Z, U, V ajouté à l'entrée (prédiction du pas de temps suivant).
- **Pas de dimension batch** dans le modèle ; utiliser `jax.vmap` pour le batching dans la boucle d'entraînement.
- Pipeline de données : XArray → numpy → JAX (pas de tf.data/TFRecord).
- Checkpoints : Orbax `PyTreeCheckpointer`.

## Fichiers modifiés (migration TF → JAX)
- `pifocast/model.py` : Nouveau `PifoModel` Flax (message passing avec `jax.ops.segment_sum`).
- `pifocast/data.py` : Plus de TF/TF-GNN ; utilise uniquement numpy + xarray.
- `pifocast/mlputils.py` : `MLP` comme `flax.linen.Module` (`build_mlp` wrapper conservé).
- `pifocast/LatLonGrid.py` : `tf.convert_to_tensor` → `jnp.array`.
- `pifocast/__init__.py` : Exports mis à jour.
- `pifocast-graph.ipynb` : Boucle d'entraînement Flax/Optax/Orbax.
- `pifocast-dataset.ipynb` : Pas de TF/TF-GNN ; sauvegarde `.npy`.
- `pyproject.toml` : `tensorflow`/`tensorflow-gnn`/`tf-keras` → `jax`/`jaxlib`/`flax`/`optax`/`orbax`.
- `PifoEncodeProcessDecode.py` : Supprimé (fusionné dans `model.py` ; archivé dans `src-archive/`).
